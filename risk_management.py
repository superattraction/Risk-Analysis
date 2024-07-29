from TOPSIS.topsis import TopsisRun
from SkylineQuery.Skq import SkylineAnalysis
import numpy as np
import json
import os

def topsis_dataset():
    file_path = './mnt/data/weight_result.json'
    numeric_columns = ['final_weighted_score', 'star', 'emp']
    weights = np.array([0.4, 0.3, 0.3])
    
    if not os.path.exists(file_path):
        print(f"파일 경로를 찾을 수 없습니다: {file_path}")
    else:
        topsis = TopsisRun(file_path, numeric_columns, weights)
        topsis.run('topsis_result2.json')


def SkylineQuery_dataset():
    skylineQuery = SkylineAnalysis('./mnt/data/weight_result.json')
    skylineQuery.run_analysis()

if __name__ == '__main__':
    topsis_dataset()
    SkylineQuery_dataset()

