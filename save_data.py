import arviz as az
import numpy as np
batch_list = []
for i in range(14):
    current_batch = az.from_netcdf("/mnt/lustre/tianli/slice_hmc/slice_F606"+str(i)+".nc")
    print('read', "/mnt/lustre/tianli/slice_hmc/slice_F606"+str(i)+".nc")
    batch_list.append(current_batch)
inf_data_pixel = az.concat(*batch_list, dim='draw')

out_path = "/mnt/lustre/tianli/slice_hmc/inf_data_pixel_F606_concat.nc"
inf_data_pixel.to_netcdf(out_path)
print("saved:", out_path)