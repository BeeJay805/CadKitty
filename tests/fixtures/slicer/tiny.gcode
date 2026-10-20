; ForgeMind test fixture: a 2-layer part, 10x10 mm external-perimeter square per layer, with a
; small support block on layer 1. Absolute E (M82), matching real PrusaSlicer output. Hand-authored
; with KNOWN geometry so gcode_facts parses to predictable values (see tests/unit/test_gcode_facts).
M140 S60
M104 S210
M82
G21
G90
;LAYER_CHANGE
;Z:0.2
;TYPE:Support material
G0 X20 Y20
G1 X22 Y20 E0.30
G1 X22 Y22 E0.60
;TYPE:External perimeter
G0 X0 Y0
G1 X10 Y0 E1.10
G1 X10 Y10 E1.60
G1 X0 Y10 E2.10
G1 X0 Y0 E2.60
;TYPE:Internal infill
G1 X5 Y5 E3.00
;LAYER_CHANGE
;Z:0.4
;TYPE:External perimeter
G0 X0 Y0
G1 X10 Y0 E3.50
G1 X10 Y10 E4.00
G1 X0 Y10 E4.50
G1 X0 Y0 E5.00
