*SENSE:Minimize
NAME          MODEL
ROWS
 N  OBJ
 L  C0000000
 L  C0000001
 G  C0000002
 L  C0000003
COLUMNS
    MARK      'MARKER'                 'INTORG'
    X0000000  C0000001   1.000000000000e+00
    X0000000  C0000002   1.000000000000e+00
    X0000000  C0000003   1.000000000000e+00
    MARK      'MARKER'                 'INTEND'
    MARK      'MARKER'                 'INTORG'
    X0000001  C0000001  -1.000000000000e+00
    X0000001  C0000002  -1.000000000000e+00
    MARK      'MARKER'                 'INTEND'
    MARK      'MARKER'                 'INTORG'
    X0000002  C0000000   1.000000000000e+00
    X0000002  C0000001  -1.000000000000e+00
    X0000002  C0000002   1.000000000000e+00
    X0000002  C0000003  -1.000000000000e+00
    X0000002  OBJ        1.000000000000e+00
    MARK      'MARKER'                 'INTEND'
RHS
    RHS       C0000000   0.000000000000e+00
    RHS       C0000001  -1.000000000000e+00
    RHS       C0000002  -1.000000000000e+00
    RHS       C0000003   1.000000000000e+00
BOUNDS
 LO BND       X0000000  -1.141855239502e+09
 UP BND       X0000000   1.214860776427e+09
 LO BND       X0000001  -2.342500371752e+08
 UP BND       X0000001   1.925956500845e+09
 FR BND       X0000002
ENDATA
