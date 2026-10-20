"""Stripe billing for CadKitty: subscription checkout, customer portal, and the credit-granting
webhook.

INERT until configured: the routes return 503 unless STRIPE_SECRET_KEY + STRIPE_PRICE_PRO +
SUPABASE_SERVICE_ROLE_KEY (+ STRIPE_WEBHOOK_SECRET for the webhook) are set, so the app runs fine
before billing is wired.

SECURITY:
  - The webhook verifies the Stripe signature before trusting any event; without that, anyone could
    POST a fake `invoice.paid` and grant themselves credits.
  - Every credit/plan write uses the Supabase SERVICE ROLE (a user can never write their own balance;
    RLS only lets them read it). The user token is verified before any spend-side action.
  - Grants are idempotent on the Stripe event id (Stripe retries webhooks). A sequential
    retry is short-circuited by _event_already_processed(); a CONCURRENT duplicate delivery is
    caught by the partial unique index credit_ledger_stripe_reason_uniq on
    credit_ledger(reason) where reason like 'stripe:%' -- the losing grant_credits() hits the
    unique violation and the whole call rolls back, so a replay never double-grants. (Both layers
    are needed: under multiple workers the Python check alone races -- audit H5.)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO", "")
STRIPE_PRICE_MAX = os.environ.get("STRIPE_PRICE_MAX", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
PRO_MONTHLY_CREDITS = int(os.environ.get("PRO_MONTHLY_CREDITS", "150"))
MAX_MONTHLY_CREDITS = int(os.environ.get("MAX_MONTHLY_CREDITS", "500"))
APP_URL = os.environ.get("APP_URL", "https://cadkitty.com").rstrip("/")


def _checkout_configured() -> bool:
    return bool(
        STRIPE_SECRET_KEY and STRIPE_PRICE_PRO and SUPABASE_SERVICE_ROLE_KEY and SUPABASE_URL
    )


def _stripe() -> Any:
    import stripe

    stripe.api_key = STRIPE_SECRET_KEY
    return stripe


# --- Supabase admin (service role; bypasses RLS) -------------------------------------------------


def _admin_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "content-type": "application/json",
    }


def _token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="sign in")
    return authorization.split(" ", 1)[1].strip()


def _user_from_token(token: str) -> tuple[str, str]:
    """(user_id, email) for a valid Supabase token, else 401."""
    try:
        r = httpx.get(
            f"{SUPABASE_URL}/auth/v1/user",
            # /auth/v1/user only needs the public anon key as apikey (same as server/app.py);
            # don't send the high-privilege service-role key on a plain token lookup (SECURITY.md #7).
            # Fall back to the service-role key if the anon key isn't configured, to avoid breakage.
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY,
            },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="could not verify session") from exc
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="invalid or expired session")
    d = r.json()
    return d["id"], d.get("email") or ""


def _account_by(field: str, value: str) -> dict[str, Any] | None:
    try:
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/accounts",
            params={field: f"eq.{value}", "select": "*"},
            headers=_admin_headers(),
            timeout=10.0,
        )
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    rows = r.json()
    return rows[0] if rows else None


def _patch_account(user_id: str, patch: dict[str, Any]) -> None:
    httpx.patch(
        f"{SUPABASE_URL}/rest/v1/accounts",
        params={"user_id": f"eq.{user_id}"},
        headers=_admin_headers(),
        json=patch,
        timeout=10.0,
    )


def _grant(user_id: str, amount: int, reason: str) -> None:
    httpx.post(
        f"{SUPABASE_URL}/rest/v1/rpc/grant_credits",
        headers=_admin_headers(),
        json={"p_user_id": user_id, "p_amount": amount, "p_reason": reason},
        timeout=10.0,
    )


def _event_already_processed(event_id: str) -> bool:
    """True if a ledger entry for this Stripe event exists (idempotency for grants)."""
    try:
        r = httpx.get(
            f"{SUPABASE_URL}/rest/v1/credit_ledger",
            params={"reason": f"eq.stripe:{event_id}", "select": "id", "limit": "1"},
            headers=_admin_headers(),
            timeout=10.0,
        )
    except httpx.HTTPError:
        return False
    return r.status_code == 200 and bool(r.json())


def _iso(ts: int | None) -> str | None:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat() if ts else None


def _plan_credits_for_price(price_id: str | None) -> tuple[str, int]:
    """Map a Stripe price to (plan, monthly credits). Unknown prices default to Pro."""
    if price_id and STRIPE_PRICE_MAX and price_id == STRIPE_PRICE_MAX:
        return "max", MAX_MONTHLY_CREDITS
    return "pro", PRO_MONTHLY_CREDITS


class CheckoutBody(BaseModel):
    plan: str = "pro"  # "pro" | "max"


# --- routes --------------------------------------------------------------------------------------


def create_billing_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/billing/checkout")
    def checkout(
        body: CheckoutBody | None = None, authorization: str | None = Header(default=None)
    ) -> dict[str, str]:
        if not _checkout_configured():
            raise HTTPException(status_code=503, detail="billing not configured")
        plan = body.plan if body else "pro"
        if plan == "max":
            if not STRIPE_PRICE_MAX:
                raise HTTPException(status_code=503, detail="max plan not configured")
            price = STRIPE_PRICE_MAX
        else:
            price = STRIPE_PRICE_PRO
        user_id, email = _user_from_token(_token(authorization))
        acct = _account_by("user_id", user_id)
        customer_id = acct.get("stripe_customer_id") if acct else None
        stripe = _stripe()
        if not customer_id:
            customer = stripe.Customer.create(email=email, metadata={"user_id": user_id})
            customer_id = customer.id
            _patch_account(user_id, {"stripe_customer_id": customer_id})
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price, "quantity": 1}],
            success_url=f"{APP_URL}/app?upgraded=1",
            cancel_url=f"{APP_URL}/pricing",
            metadata={"user_id": user_id, "plan": plan},
            allow_promotion_codes=True,
        )
        return {"url": session.url}

    @router.post("/api/billing/portal")
    def portal(authorization: str | None = Header(default=None)) -> dict[str, str]:
        if not _checkout_configured():
            raise HTTPException(status_code=503, detail="billing not configured")
        user_id, _ = _user_from_token(_token(authorization))
        acct = _account_by("user_id", user_id)
        customer_id = acct.get("stripe_customer_id") if acct else None
        if not customer_id:
            raise HTTPException(status_code=400, detail="no subscription to manage")
        stripe = _stripe()
        session = stripe.billing_portal.Session.create(
            customer=customer_id, return_url=f"{APP_URL}/app"
        )
        return {"url": session.url}

    @router.post("/api/stripe/webhook")
    async def webhook(
        request: Request,
        stripe_signature: str | None = Header(default=None, alias="stripe-signature"),
    ) -> dict[str, bool]:
        if not (STRIPE_WEBHOOK_SECRET and SUPABASE_SERVICE_ROLE_KEY and SUPABASE_URL):
            raise HTTPException(status_code=503, detail="webhook not configured")
        payload = await request.body()
        stripe = _stripe()
        try:
            event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
        except Exception as exc:  # signature mismatch / malformed
            raise HTTPException(status_code=400, detail="invalid signature") from exc
        _handle_event(event)
        return {"received": True}

    return router


def _handle_event(event: Any) -> None:
    etype = event["type"]
    obj = event["data"]["object"]
    event_id = event["id"]

    if etype in ("invoice.paid", "invoice.payment_succeeded"):
        if _event_already_processed(event_id):
            return  # idempotent: Stripe retried a grant we already applied
        acct = _account_by("stripe_customer_id", obj.get("customer", ""))
        if not acct:
            return
        lines = (obj.get("lines") or {}).get("data") or []
        price_id = (lines[0].get("price") or {}).get("id") if lines else None
        plan, credits = _plan_credits_for_price(price_id)
        _grant(acct["user_id"], credits, f"stripe:{event_id}")
        period_end = _iso((lines[0].get("period") or {}).get("end")) if lines else None
        _patch_account(
            acct["user_id"],
            {"plan": plan, "subscription_status": "active", "current_period_end": period_end},
        )

    elif etype == "customer.subscription.deleted":
        acct = _account_by("stripe_customer_id", obj.get("customer", ""))
        if acct:
            _patch_account(acct["user_id"], {"plan": "free", "subscription_status": "canceled"})

    elif etype == "customer.subscription.updated":
        acct = _account_by("stripe_customer_id", obj.get("customer", ""))
        if acct:
            status = obj.get("status", "")
            items = (obj.get("items") or {}).get("data") or []
            price_id = (items[0].get("price") or {}).get("id") if items else None
            plan, _credits = _plan_credits_for_price(price_id)
            _patch_account(
                acct["user_id"],
                {
                    "subscription_status": status,
                    "plan": plan if status in ("active", "trialing") else "free",
                    "current_period_end": _iso(obj.get("current_period_end")),
                },
            )


__all__ = ["create_billing_router"]
