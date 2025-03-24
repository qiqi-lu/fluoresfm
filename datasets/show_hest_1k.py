import scanpy

anndata = scanpy.read_h5ad(filename="HEST-1k\zip\st\TENX99.h5ad")
print(anndata.to_df())
