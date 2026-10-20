# Recorded instructions

These are recorded author instructions, not physical qualification or acceptance.
Separate print-package paths may refer to files outside this showcase.

## Print

This is a machining design draft. No separate print files, plate orientation, printer settings, supports or G-code are available. Print preparation was not requested and is not required for this delivery. The saved combined STL is not watertight and is excluded from the selected package. Do not use it as a print-ready assembly.

The following STEP geometry sources are available. STEP files describe solid shapes for CAD review. These are prospective package-relative paths; package capture and download readback are not established. Check the final manifest.json for actual package contents.

Made-part designs and required quantities:
- Aluminum inline-four block with open bearing saddles, 1: make/definition_engine_block.step.
- Steel four-throw crankshaft with integral handwheel, 1: make/definition_crankshaft.step.
- Steel camshaft with four eccentric demonstration lobes, 1: make/definition_camshaft.step.
- Flanged bronze crankshaft bushings with oil feeds, 2: make/definition_crankshaft_bushing.step.
- Flanged bronze camshaft bushings with oil feeds, 2: make/definition_camshaft_bushing.step.
- Aluminum pistons with rod slots and wrist-pin oil feeds, 4: make/definition_piston.step.
- Aluminum connecting-rod upper bodies with open big-end saddles, 4: make/definition_connecting_rod.step.
- Aluminum crankcase cover with lower bearing supports, 1: make/definition_crankcase_service_cover.step.
- Aluminum camshaft cover with bearing supports and oil feeds, 1: make/definition_camshaft_service_cover.step.

These quantities describe modeled instances, not manufactured parts. assembly/part.step provides the overall assembly for review. Project geometry retains its original CAD coordinates, not a print-bed placement.

The prospective metadata files are document.json, production.json, purchasing.json, sources.json, engineering-evidence.json, manifest.json and README.md. document.json is the editable native design document. Separate purchased-pin and cover-screw STEP files are excluded from this package.

## Installation

Use this sequence for assembly planning only. It is not a verified installation procedure. Complete the missing design work listed below before physical assembly.

The proposed purchased hardware is four nominal steel wrist pins and ten nominal M3 by 8 slotted steel cover screws. The cover layout uses six bottom screws and four side screws. Supplier products, pin tolerances, screw engagement and tightening torque remain unspecified. Modeled screw holes and shanks are nominal thread envelopes, not a thread-manufacturing specification.

Proposed shop equipment includes a mill, lathe, drilling equipment, offset shaft fixtures, boring or reaming tools, and deburring tools. Proposed inspection tools include micrometers, bore gauges and a dial indicator. Assembly would need supported workholding and a screwdriver matched to the selected screws. Exact tooling depends on the completed manufacturing specification.

Proposed sequence, subject to an assembly-access review:
1. Check the parts against the completed manufacturing drawings.
2. Remove burrs and clean all oil passages.
3. Match each service cover to the block before finishing its bearing seats.
4. Finish each connecting-rod big-end bore with its matching cap installed.
5. Join each piston to its rod with a wrist pin and the specified retention parts.
6. Install the crankshaft and its two bushings through the intended bottom access.
7. Place the piston-and-rod units in their cylinders.
8. Attach the rods around the crankpins with their matching caps and specified screws.
9. Install the camshaft and its two bushings through the intended side access.
10. Align the bushing oil holes with their supply passages.
11. Install the specified shaft retention and bushing anti-rotation parts.
12. Fit both covers with the confirmed cover screws.
13. Apply the specified lubricant before the controlled hand-turning test.

The declared process uses conventional milling, turning and drilling. Aluminum 6061-T6 and low-carbon machining steel are proposed choices. The bearing-bronze grade still needs selection. Material labels and preview colors do not verify actual materials.

## Use and removal

Use the current files for design review and manufacturing planning only. Do not operate this incomplete assembly. No combustion qualification is requested or provided.

After design completion and physical validation, the intended use is slow hand operation through the integral crankshaft handwheel. Keep fingers clear of moving joints. Stop if the mechanism binds; do not force the handwheel. The camshaft drive and demonstration behavior remain unspecified, so coordinated crankshaft-to-camshaft motion must not be assumed.

Proposed service sequence, subject to access verification:
1. Stop all movement.
2. Support the block and the shaft served by the cover.
3. Remove the applicable cover screws.
4. Withdraw the cover without loading its bearing supports sideways.
5. Keep each cover with its matched block seats.
6. Follow the completed retention-release procedure before removing a shaft or wrist pin.

Both covers include bearing supports. Removing a cover can remove shaft support. A complete bushing-replacement procedure still needs confirmation of extraction clearance and access. Do not hammer parts out or pry against finished bearing surfaces.

## Unresolved

The draft lacks four matching connecting-rod caps and their screws. Positive shaft retention, wrist-pin retention and bushing anti-rotation remain incomplete. These omissions prevent completion of the intended working mechanism.

The saved STEP geometry is valid and contains 31 solids. Its measured envelope is X 172, Y 83.0000001, Z 95 mm, within the requested limits. Those three saved envelope checks pass. The other 18 requirements remain unverified; no current failed requirement checks are listed. The overall requirements are not satisfied.

The captured plan and selected CAD revision are aligned. Construction-feature presence does not verify the intended mechanical effects. No motion rig or full-cycle collision evidence is supplied. Nominal interface checks do not establish retention or hand-turnability.

Manufacturing access, fits, tolerances, surface finishes and thread details need review. Lubricant selection, oil distribution to all moving joints and passage continuity need confirmation. Physical fit, strength, safe operation and successful manufacture or printing remain unverified. A quantitative hand-turning criterion and physical test results are absent.

This instruction review records review of the supplied saved facts only. It does not complete missing geometry, establish product approval or confirm delivery of the final package. Detailed check identities and evidence belong in the accompanying engineering report.
