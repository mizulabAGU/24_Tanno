import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

class DataVisualizer:
    """
    データを読み込み、処理し、箱ひげ図を生成するクラス。
    """
    def __init__(self, directory):
        """
        初期化メソッド。
        
        Args:
            directory (str): Excelファイルが保存されているディレクトリのパス。
        """
        self.directory = directory
        # Excelファイルのパターンを設定
        self.file_pattern = os.path.join(directory, "simulation_results_rate_*.xlsx")
        # データを格納する辞書
        self.data = {}
        # パーセンテージのリスト
        self.percentages = []

    def load_data(self):
        """
        指定されたディレクトリからExcelファイルを読み込み、データを処理するメソッド。
        
        Raises:
            FileNotFoundError: 該当するExcelファイルが見つからなかった場合に発生。
        """
        # ファイルパターンにマッチするファイルを取得
        files = glob.glob(self.file_pattern)
        if not files:
            raise FileNotFoundError("該当するExcelファイルが見つかりません。")
        
        # ファイルをソートして処理
        for file in sorted(files):
            try:
                # ファイル名からパーセンテージを抽出
                percentage = self.extract_percentage(file)
                # Excelファイルを読み込み、必要な列を取得
                df = pd.read_excel(file, usecols=[0, 5], header=None)
                df.columns = ['A', 'F']
                # A列でグループ化し、F列の合計を計算
                sums = df.groupby('A')['F'].sum().reindex(range(1, 101), fill_value=0).tolist()
                # 計算結果をデータ辞書に保存
                self.data[percentage] = sums
                self.percentages.append(percentage)
            except Exception as e:
                print(f"ファイル {file} の処理中にエラーが発生しました: {e}")

    def extract_percentage(self, filename):
        """
        ファイル名からパーセンテージを抽出するメソッド。
        
        Args:
            filename (str): ファイルのパス。
        
        Returns:
            float: 抽出したパーセンテージ。
        
        Raises:
            ValueError: パーセンテージを抽出できなかった場合に発生。
        """
        basename = os.path.basename(filename)
        try:
            # ファイル名から '_rate_' と '%' の間の文字列を抽出
            percentage_str = basename.split('_rate_')[1].split('%')[0]
            return float(percentage_str)
        except (IndexError, ValueError) as e:
            raise ValueError(f"ファイル名 '{basename}' からパーセンテージを抽出できません。") from e

    def plot_boxplot(self):
        """
        読み込んだデータを基に箱ひげ図を作成し、保存・表示するメソッド。
        """
        try:
            # パーセンテージ順にデータを並べ替え
            data_to_plot = [self.data[p] for p in sorted(self.percentages)]
            sorted_percentages = sorted(self.percentages)
            plt.figure(figsize=(12, 6))
            # 箱ひげ図を作成
            plt.boxplot(data_to_plot, positions=sorted_percentages, widths=5)
            plt.xlabel('パーセンテージ')
            plt.ylabel('F列の合計値')
            plt.title('シミュレーション結果の箱ひげ図')
            # X軸の目盛りをパーセンテージ表示に設定
            plt.xticks(sorted_percentages, [f"{p}%" for p in sorted_percentages], rotation=45)
            plt.tight_layout()
            # 箱ひげ図を画像として保存
            plt.savefig(os.path.join(self.directory, 'simulation_results_boxplot.png'))
            # 箱ひげ図を表示
            plt.show()
        except Exception as e:
            print(f"箱ひげ図の作成中にエラーが発生しました: {e}")

def main():
    """
    メイン関数。DataVisualizerクラスを使用してデータの読み込みと箱ひげ図の作成を行う。
    """
    # 現在のスクリプトが存在するディレクトリを取得
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # DataVisualizerのインスタンスを作成
    visualizer = DataVisualizer(current_directory)
    
    try:
        # データを読み込む
        visualizer.load_data()
        # 箱ひげ図をプロット
        visualizer.plot_boxplot()
    except Exception as e:
        print(f"プログラム実行中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
