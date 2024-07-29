import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import pandas as pd
import os

class TopsisRun:
    def __init__(self, json_filepath, numeric_columns, weights):
        self.json_filepath = json_filepath
        self.numeric_columns = numeric_columns
        self.weights = weights
        self.df = self.preprocess()
        
    def preprocess(self):
        """
        JSON 파일을 읽어 데이터프레임으로 변환하고 정규화할 열들을 선택함.
        """
        with open(self.json_filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        df = pd.DataFrame(data)

        # 데이터 타입 변환 (숫자형으로 변환)
        df[self.numeric_columns] = df[self.numeric_columns].apply(pd.to_numeric, errors='coerce')

        # 결측값 확인 및 처리 (필요한 경우)
        df = df.dropna(subset=self.numeric_columns)

        return df

    def topsis_algorithm(self):
        """
        TOPSIS 알고리즘을 사용하여 데이터를 정규화하고, 가중치를 적용한 후, 
        이상적인 해와 반이상적인 해를 계산하여 순위를 매김.
        """
        # Step 1: Normalize the data
        scaler = MinMaxScaler()
        normalized_df = scaler.fit_transform(self.df[self.numeric_columns])

        # Step 2: Apply weights
        weighted_df = normalized_df * self.weights

        # Step 3: Identify ideal and anti-ideal solutions
        ideal_solution = np.max(weighted_df, axis=0)
        anti_ideal_solution = np.min(weighted_df, axis=0)

        # Step 4: Calculate the distance to the ideal and anti-ideal solutions
        distance_to_ideal = np.sqrt(np.sum((weighted_df - ideal_solution) ** 2, axis=1))
        distance_to_anti_ideal = np.sqrt(np.sum((weighted_df - anti_ideal_solution) ** 2, axis=1))

        # Step 5: Calculate the similarity to the ideal solution
        similarity_to_ideal = distance_to_anti_ideal / (distance_to_ideal + distance_to_anti_ideal)

        # Step 6: Rank the solutions
        ranking = np.argsort(similarity_to_ideal)[::-1]
        return ranking

    def rank(self):
        """
        전처리된 데이터를 받아 TOPSIS 알고리즘을 적용하고, 
        상위 10개와 하위 10개의 후보를 선택하여 최적의 후보를 출력함.
        """
        try:
            # TOPSIS 알고리즘 적용
            ranking = self.topsis_algorithm()

            # 베스트 후보와 워스트 후보 선택
            best_indices = ranking[:10]  # 상위 10개 후보 선택
            worst_indices = ranking[-10:]  # 하위 10개 후보 선택

            # 최종 결과 준비
            ranked_best_candidates = [
                {"rank": i + 1, **self.df.iloc[idx].to_dict()}
                for i, idx in enumerate(best_indices)
            ]
            ranked_worst_candidates = [
                {"rank": len(worst_indices) - i, **self.df.iloc[idx].to_dict()}
                for i, idx in enumerate(worst_indices)
            ]

            # 최적 후보를 하나의 요소로 가진 리스트로 준비
            one_prime_candidate = [{"rank": 1, **self.df.iloc[best_indices[0]].to_dict()}]

            # 결과 데이터를 JSON으로 변환하여 저장
            result_data = {
                "prime_candidate": one_prime_candidate,
                "ranked_best_candidates": ranked_best_candidates,
                "ranked_worst_candidates": ranked_worst_candidates
            }

            return result_data
        
        except ValueError as ve:
            print(f"값 오류 발생: {ve}")
            return None
        except Exception as e:
            print(f"오류 발생: {e}")
            return None
    
    def save_result(self, result_data, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

    def run(self, output_file):
        result_data = self.rank()
        if result_data:
            self.save_result(result_data, output_file)
            print(f"JSON 파일 저장 완료: {output_file}")
        else:
            print("결과 데이터가 없습니다.")


