*SENSE:Minimize
NAME          MODEL
ROWS
 N  OBJ
 L  C0000000
 L  C0000001
 G  C0000002
 G  C0000003
 G  C0000004
 G  C0000005
COLUMNS
    MARK      'MARKER'                 'INTORG'
    X0000000  C0000001   1.000000000000e+00
    X0000000  C0000002   1.000000000000e+00
    X0000000  C0000003   1.000000000000e+00
    MARK      'MARKER'                 'INTEND'
    MARK      'MARKER'                 'INTORG'
    X0000001  C0000001  -1.000000000000e+00
    X0000001  C0000002  -1.000000000000e+00
    X0000001  C0000004   1.000000000000e+00
    MARK      'MARKER'                 'INTEND'
    MARK      'MARKER'                 'INTORG'
    X0000002  C0000001  -2.331468351713e-15
    X0000002  C0000002  -2.331468351713e-15
    X0000002  C0000005   1.000000000000e+00
    MARK      'MARKER'                 'INTEND'
    MARK      'MARKER'                 'INTORG'
    X0000003  C0000000   1.000000000000e+00
    X0000003  C0000001  -1.000000000000e+00
    X0000003  C0000002   1.000000000000e+00
    X0000003  OBJ        1.000000000000e+00
    MARK      'MARKER'                 'INTEND'
RHS
    RHS       C0000000   0.000000000000e+00
    RHS       C0000001  -1.999997598897e+00
    RHS       C0000002  -1.999997598897e+00
    RHS       C0000003  -2.147483648000e+09
    RHS       C0000004  -2.147483648000e+09
    RHS       C0000005  -2.147483648000e+09
BOUNDS
 LO BND       X0000000  -1.081966308049e+08
 UP BND       X0000000   1.378709232598e+09
 LO BND       X0000001  -1.822177646058e+09
 UP BND       X0000001   1.653347412198e+09
 LO BND       X0000002  -9.597500026911e+08
 UP BND       X0000002   1.358426701414e+09
 FR BND       X0000003
ENDATA
