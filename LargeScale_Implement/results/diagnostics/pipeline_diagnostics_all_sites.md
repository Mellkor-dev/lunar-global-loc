# Pipeline localization diagnostics

Errors are horizontal radial errors in metres. An em dash means no feature-derived pose was available.
Odometry position is used only as evaluation truth.

| Resolution | Site | DARCES status | DARCES x | DARCES y | DARCES error | RANSAC status | RANSAC x | RANSAC y | RANSAC error | D→R | MOGA status | MOGA x | MOGA y | MOGA error | R→M | Best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10m | 1 | solution | -446.31 | -159.83 | 474.06 | solution | -446.31 | -159.83 | 474.06 | unchanged | solution | -448.45 | -160.89 | 476.44 | worsened | DARCES |
| 10m | 2 | solution | -546.18 | -554.97 | 786.85 | solution | -546.18 | -554.97 | 786.85 | unchanged | solution | -536.76 | -556.46 | 781.24 | improved | MOGA |
| 10m | 3 | solution | 22.53 | -9.69 | 1.46 | solution | 22.53 | -9.69 | 1.46 | unchanged | solution | 22.61 | -12.34 | 2.50 | worsened | DARCES |
| 10m | 4 | solution | 699.78 | 633.92 | 941.42 | solution | 699.78 | 633.92 | 941.42 | unchanged | solution | 696.25 | 632.68 | 938.05 | improved | MOGA |
| 10m | 5 | solution | -471.32 | -189.09 | 511.13 | solution | -471.32 | -189.09 | 511.13 | unchanged | solution | -472.14 | -189.68 | 512.09 | worsened | DARCES |
| 10m | 6 | solution | -5.60 | -58.44 | 4.67 | solution | -5.60 | -58.44 | 4.67 | unchanged | solution | -0.55 | -59.61 | 0.72 | improved | MOGA |
| 10m | 7 | solution | 710.10 | 606.17 | 996.02 | minimal_unverified | 710.10 | 606.17 | 996.02 | unchanged | solution | 710.74 | 606.75 | 996.89 | worsened | DARCES |
| 10m | 8 | skipped_insufficient_features | — | — | — | darces_unavailable | — | — | — | not_comparable | feature_pose_unavailable | — | — | — | not_comparable | none |
| 10m | 9 | skipped_insufficient_features | — | — | — | darces_unavailable | — | — | — | not_comparable | feature_pose_unavailable | — | — | — | not_comparable | none |
| 10m | 10 | no_solution | — | — | — | darces_unavailable | — | — | — | not_comparable | feature_pose_unavailable | — | — | — | not_comparable | none |
| 10m | 11 | solution | -333.11 | -201.21 | 310.15 | solution | -333.11 | -201.21 | 310.15 | unchanged | solution | -329.11 | -199.01 | 305.87 | improved | MOGA |
| 10m | 12 | solution | -287.74 | -226.98 | 265.80 | solution | -287.74 | -226.98 | 265.80 | unchanged | solution | -288.15 | -221.95 | 265.26 | improved | MOGA |
| 10m | 13 | solution | -482.96 | -434.02 | 509.64 | minimal_unverified | -482.96 | -434.02 | 509.64 | unchanged | solution | -485.46 | -433.90 | 511.81 | worsened | DARCES |
| 10m | 14 | solution | -175.19 | -19.26 | 240.31 | solution | -175.19 | -19.26 | 240.31 | unchanged | solution | -168.95 | -21.78 | 234.96 | improved | MOGA |
| 10m | 15 | solution | -539.76 | -530.66 | 749.70 | solution | -539.76 | -530.66 | 749.70 | unchanged | solution | -542.02 | -528.74 | 749.91 | worsened | DARCES |
| 10m | 16 | solution | -201.39 | -255.35 | 72.33 | solution | -201.39 | -255.35 | 72.33 | unchanged | solution | -191.84 | -249.05 | 76.10 | worsened | DARCES |
| 10m | 17 | no_solution | — | — | — | darces_unavailable | — | — | — | not_comparable | feature_pose_unavailable | — | — | — | not_comparable | none |
| 10m | 18 | solution | 944.45 | 168.80 | 1169.86 | minimal_unverified | 944.45 | 168.80 | 1169.86 | unchanged | solution | 944.50 | 168.80 | 1169.91 | worsened | DARCES |
| 10m | 19 | solution | -617.87 | -485.44 | 487.71 | solution | -617.87 | -485.44 | 487.71 | unchanged | solution | -625.21 | -484.65 | 494.28 | worsened | DARCES |
| 10m | 20 | solution | -592.67 | -478.00 | 422.54 | solution | -592.67 | -478.00 | 422.54 | unchanged | solution | -595.53 | -476.97 | 424.91 | worsened | DARCES |
| 10m | 21 | no_solution | — | — | — | darces_unavailable | — | — | — | not_comparable | feature_pose_unavailable | — | — | — | not_comparable | none |
| 10m | 22 | no_solution | — | — | — | darces_unavailable | — | — | — | not_comparable | feature_pose_unavailable | — | — | — | not_comparable | none |
| 10m | 23 | solution | -249.99 | -378.86 | 2.69 | solution | -249.99 | -378.86 | 2.69 | unchanged | solution | -250.11 | -374.10 | 4.52 | worsened | DARCES |
| 10m | 24 | no_solution | — | — | — | darces_unavailable | — | — | — | not_comparable | feature_pose_unavailable | — | — | — | not_comparable | none |
| 10m | 25 | no_solution | — | — | — | darces_unavailable | — | — | — | not_comparable | feature_pose_unavailable | — | — | — | not_comparable | none |
| 10m | 26 | no_solution | — | — | — | darces_unavailable | — | — | — | not_comparable | feature_pose_unavailable | — | — | — | not_comparable | none |
| 10m | 27 | solution | -355.76 | -305.13 | 33.96 | solution | -355.76 | -305.13 | 33.96 | unchanged | solution | -355.94 | -302.70 | 36.38 | worsened | DARCES |
| 10m | 28 | solution | -313.31 | -252.58 | 134.11 | solution | -313.31 | -252.58 | 134.11 | unchanged | solution | -311.15 | -253.57 | 134.57 | worsened | DARCES |
