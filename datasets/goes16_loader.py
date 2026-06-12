import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import handle_nans, crop_center, normalize_channels

class GOES16ProxyDataset(Dataset):
    """
    Streams consecutive frame pairs directly from NOAA's public GOES-16 S3 bucket.

    Product: 'ABI-L2-CMIPC' (Cloud & Moisture Imagery - CONUS standard projection)
    Bands: 'C09' (6.9um Mid-Level Water Vapor) or 'C14' (11.2um Longwave Thermal IR)
    """
    def __init__( self, product='ABI-L2-CMIPC', band='C09', year=2025, day_of_year=150, hour=14,
    crop_size=512, normalization_method='minmax'):

        self.bucket_name = 'noaa-goes16'
        self.product = product
        self.band = band
        self.crop_size = crop_size
        self.normalization_method = normalization_method

        # Configure anonymous public access to bypass mandatory AWS credential steps
        self.s3 = boto3.client('s3', region_name='us-east-1',
                               config=Config(signature_version=UNSIGNED))

        # S3 path syntax structure: Product/Year/Day_of_Year/Hour/
        self.prefix = f"{product}/{year}/{day_of_year:03d}/{hour:02d}/"
        self.file_list = self._get_s3_file_list()

        if len(self.file_list) < 2:
            raise ValueError(f"Not enough GOES-16 files found for {self.prefix}")

    def _get_s3_file_list(self):
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix)
        if 'Contents' not in response:
            raise FileNotFoundError(f"No files available matching path: s3://{self.bucket_name}/{self.prefix}")

        # Match only files containing our desired band descriptor suffix and NetCDF files.
        files = [obj['Key'] for obj in response['Contents']
                 if f"M6{self.band}" in obj['Key'] and obj['Key'].endswith('.nc')]
        return sorted(files)

    def _download_and_parse(self, s3_key):
        local_filename = os.path.basename(s3_key)
        if not os.path.exists(local_filename):
            print(f"Streaming {local_filename} down to disk runtime...")
            self.s3.download_file(self.bucket_name, s3_key, local_filename)

        with nc.Dataset(local_filename, 'r') as rootgrp:
            data_matrix = np.array(rootgrp.variables['CMI'][:], dtype=np.float32)

        data_matrix = handle_nans(data_matrix, replacement_val=0.0, method='mean')
        data_matrix = crop_center(data_matrix, crop_h=self.crop_size, crop_w=self.crop_size)
        data_matrix = normalize_channels(data_matrix, method=self.normalization_method)

        return torch.from_numpy(data_matrix).unsqueeze(0) # Shape: [1, H, W]

    def __len__(self):
        return max(0, len(self.file_list) - 1)

    def __getitem__(self, idx):
        frame_t = self._download_and_parse(self.file_list[idx])
        frame_t_next = self._download_and_parse(self.file_list[idx + 1])

        _, h, w = frame_t.shape
        mock_dem = torch.zeros((1, h, w), dtype=torch.float32)

        return frame_t, frame_t_next, mock_dem
