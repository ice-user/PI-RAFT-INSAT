import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset

class GOES16ProxyDataset(Dataset):
    """
    Streams consecutive frame sequences from NOAA's public GOES-16 S3 bucket.
    Product: 'ABI-L2-CMIPC' (Cloud & Moisture Imagery - CONUS standard projection)
    Bands: 'C09' (6.9um Mid-Level Water Vapor) or 'C14' (11.2um Longwave Thermal IR)
    """
    def __init__(self, product='ABI-L2-CMIPC', band='C09', year=2025, day_of_year=150, hour=14, sequence_length=2):
        self.bucket_name = 'noaa-goes16'
        self.product = product
        self.band = band
        self.sequence_length = sequence_length

        # Configure anonymous public access to bypass mandatory AWS credential steps
        self.s3 = boto3.client('s3', region_name='us-east-1',
                               config=Config(signature_version=UNSIGNED))

        # S3 Path syntax structure: Product/Year/Day_of_Year/Hour/
        self.prefix = f"{product}/{year}/{day_of_year:03d}/{hour:02d}/"
        self.file_list = self._get_s3_file_list()

    def _get_s3_file_list(self):
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.prefix)
        if 'Contents' not in response:
            raise FileNotFoundError(f"No files available matching path: s3://{self.bucket_name}/{self.prefix}")

        # Match only files containing our desired band descriptor suffix
        files = [obj['Key'] for obj in response['Contents'] if f"M6{self.band}" in obj['Key']]
        return sorted(files)

    def _download_and_parse(self, s3_key):
        local_filename = s3_key.split('/')[-1]
        if not os.path.exists(local_filename):
            print(f"Streaming {local_filename} down to disk runtime...")
            self.s3.download_file(self.bucket_name, s3_key, local_filename)

        # Parse matrix metadata
        with nc.Dataset(local_filename, 'r') as rootgrp:
            data_matrix = np.array(rootgrp.variables['CMI'][:], dtype=np.float32)
            data_matrix = np.nan_to_num(data_matrix, nan=0.0)

        h, w = data_matrix.shape
        data_matrix = data_matrix[h//2-256 : h//2+256, w//2-256 : w//2+256]

        return torch.from_numpy(data_matrix).unsqueeze(0) # Shape: [1, H, W]

    def __len__(self):
        return max(0, len(self.file_list) - self.sequence_length + 1)

    def __getitem__(self, idx):
        frames = [
            self._download_and_parse(self.file_list[idx + i])
            for i in range(self.sequence_length)
        ]

        # Terrain Context layer (DEM mapping simulation) matching image geometry
        _, h, w = frames[0].shape
        mock_dem = torch.zeros((1, h, w), dtype=torch.float32)

        return torch.stack(frames, dim=0), mock_dem
