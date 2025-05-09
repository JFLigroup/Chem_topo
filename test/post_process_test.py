from chem_topo.post_process import AlphaComplexAnalyzer
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chem_topo.post_process import AlphaComplexAnalyzer

if __name__ == "__main__":
    index = int(sys.argv[1]) 
    folder = './example/'
    file_name = 'PtKOH'
    analyzer = AlphaComplexAnalyzer(folder_path=folder, file_name=file_name)
    analyzer.run(task_index=index)