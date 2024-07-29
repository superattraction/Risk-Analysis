import numpy as np
import pandas as pd
from rtree import index
import heapq
import torch
import json

class SkylineAnalysis:
    def __init__(self, json_filepath):
        self.json_filepath = json_filepath
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.df = self.load_json()
        self.adjust_values()
        self.ps_tensor, self.s_job_avg_tensor = self.to_tensor()
        self.rtree_index = self.create_rtree()
    
    def load_json(self):
        with open(self.json_filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    
    def adjust_values(self):
        self.df['final_weighted_score'] = 100 - self.df['final_weighted_score']
        self.df['emp'] = 100 - self.df['emp']
    
    def to_tensor(self):
        ps_tensor = torch.tensor(self.df['final_weighted_score'].values, dtype=torch.float32).to(self.device)
        s_job_avg_tensor = torch.tensor(self.df['emp'].values, dtype=torch.float32).to(self.device)
        return ps_tensor, s_job_avg_tensor
    
    def create_rtree(self):
        p = index.Property()
        p.dimension = 2
        idx = index.Index(properties=p)
        for i in range(len(self.ps_tensor)):
            idx.insert(i, (self.ps_tensor[i].item(), self.s_job_avg_tensor[i].item(), self.ps_tensor[i].item(), self.s_job_avg_tensor[i].item()))
        return idx
    
    def is_dominated(self, point, skyline, threshold):
        for sk_point in skyline:
            if (sk_point[0] >= point[0] - threshold and sk_point[1] >= point[1] + threshold) and (sk_point[0] > point[0] or sk_point[1] > point[1]):
                return True
        return False
    
    def prune_candidates(self, candidates, point, threshold):
        pruned_candidates = []
        for candidate in candidates:
            if not self.is_dominated(candidate, [point], threshold):
                pruned_candidates.append(candidate)
        return pruned_candidates
    
    def compute_skyline(self):
        skyline = []
        heap = []

        for i in range(len(self.ps_tensor)):
            distance = torch.sqrt((self.ps_tensor[i] - 100)**2 + (self.s_job_avg_tensor[i] - 100)**2).item()
            heapq.heappush(heap, (distance, i))

        visited_nodes = set()

        while heap:
            _, node_id = heapq.heappop(heap)
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)

            point = (self.ps_tensor[node_id].item(), self.s_job_avg_tensor[node_id].item())
            if not self.is_dominated(point, skyline, threshold=3):
                skyline.append(point)
                candidates = []
                for i in self.rtree_index.intersection((point[0], point[1], point[0], point[1])):
                    neighbor_point = (self.ps_tensor[i].item(), self.s_job_avg_tensor[i].item())
                    if not self.is_dominated(neighbor_point, skyline, threshold=1):
                        candidates.append(neighbor_point)

                pruned_candidates = self.prune_candidates(candidates, point, threshold=3)
                for candidate in pruned_candidates:
                    candidate_indices = torch.nonzero((self.ps_tensor == candidate[0]) & (self.s_job_avg_tensor == candidate[1]), as_tuple=True)[0]
                    for candidate_index in candidate_indices:
                        distance = torch.sqrt((self.ps_tensor[candidate_index] - 100)**2 + (self.s_job_avg_tensor[candidate_index] - 100)**2).item()
                        heapq.heappush(heap, (distance, candidate_index.item()))

        return skyline
    
    def add_info_to_skyline_df(self, skyline_points):
        skyline_df = pd.DataFrame(skyline_points, columns=['final_weighted_score', 'emp'])
        skyline_df['100-final_weighted_score'] = (100 - skyline_df['final_weighted_score']).round(2)
        skyline_df['100-emp'] = (100 - skyline_df['emp']).round(2)
        return skyline_df
    
    def find_nearest_neighbors(self, skyline_points):
        used_indices = set()
        nearest_neighbors = []
        for idx, point in enumerate(skyline_points):
            nearest_neighbor_indices = list(self.rtree_index.nearest((point[0], point[1], point[0], point[1]), 4))[1:]  # 첫 번째는 자기 자신이므로 제외

            filtered_neighbors = self.df.iloc[nearest_neighbor_indices]
            filtered_neighbors = filtered_neighbors[~filtered_neighbors.index.isin(used_indices)].head(3)
            used_indices.update(filtered_neighbors.index)
            filtered_neighbors['skyline_index'] = idx
            nearest_neighbors.append(filtered_neighbors)
        return nearest_neighbors
    
    def save_json(self, skyline_df, all_neighbors_sorted, filepath):
        result_data = {
            "skyline_points": skyline_df.to_dict(orient="records"),
            "nearest_neighbors": all_neighbors_sorted.to_dict(orient="records")
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
    
    def run_analysis(self):
        skyline_points = self.compute_skyline()
        skyline_df = self.add_info_to_skyline_df(skyline_points)
        skyline_df['edu_institute'] = self.df['edu_institute'].iloc[skyline_df.index]
        skyline_df['all_reviews'] = self.df['all_reviews'].iloc[skyline_df.index]

        nearest_neighbors = self.find_nearest_neighbors(skyline_points)

        all_neighbors = pd.concat(nearest_neighbors).reset_index(drop=True)
        all_neighbors['100-final_weighted_score'] = 100 - all_neighbors['final_weighted_score']
        all_neighbors['100-emp'] = 100 - all_neighbors['emp']
        all_neighbors_sorted = all_neighbors.sort_values(by='skyline_index').reset_index(drop=True)

        # 소수점 두째자리에서 반올림
        skyline_df = skyline_df.round({'final_weighted_score': 2, 'emp': 2})
        all_neighbors_sorted = all_neighbors_sorted.round({'final_weighted_score': 2, 'emp': 2, '100-final_weighted_score': 2, '100-emp': 2})

        self.save_json(skyline_df, all_neighbors_sorted, 'skylineQuery_result1.json')

        print("\nJSON 파일 저장 완료: skylineQuery_result1.json")