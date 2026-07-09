# scripts/07_diagnose_z_outliers.py
import numpy as np

pts = np.load('sim/data/local_scan_world.npy')
z = pts[:, 2]

print(f"z percentiles: 1%={np.percentile(z,1):.3f} 5%={np.percentile(z,5):.3f} "
      f"50%={np.percentile(z,50):.3f} 95%={np.percentile(z,95):.3f} 99%={np.percentile(z,99):.3f}")

# points below global DEM's actual minimum
outliers = pts[z < -0.4]
print(f"\n{len(outliers)} points below z=-0.4 (deeper than global DEM min)")
if len(outliers) > 0:
    print("Their x range:", outliers[:,0].min(), outliers[:,0].max())
    print("Their y range:", outliers[:,1].min(), outliers[:,1].max())
    print("Sample:", outliers[:5])
