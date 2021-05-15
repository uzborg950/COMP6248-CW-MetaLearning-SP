Linked to: https://github.com/COMP6248-Reproducability-Challenge/Structured-Prediction-for-Conditional-MetaLearning-with-Deep-Models
# COMP6248-CW-MetaLearning-SP

Implements paper: ["Structured Prediction for Conditional Meta-Learning" by Wang, Ruohan Demiris, Yiannis Ciliberto, Carlo](https://arxiv.org/abs/2002.08799) 

Uses miniImageNet and tieredImageNet embeddings from https://github.com/deepmind/leo. 

### Dependecies.
- Python: Python3.9.x
- NumPy: 1.19.5
- Torch: 1.8.1
- Torchvision: 0.9.1
- tqdm: 4.60.0

### Executing benchmarks.
1. Set your n-way k-show and embedding preferences in config_flags.py 
2. Set/unset flag top_m_filtering in embs_test_runner.py 
3. Set the preferred inner module in embs_test_runner.py, function get_test_net. For example TestNets.MAMLModule1(input_len=640, n_classes=Config.NUM_OF_CLASSES) to TestNets.Module1(input_len=640, n_classes=Config.NUM_OF_CLASSES) 
5. Set the output CSV name in global variable OUTPUT_FILE_NAME in testing_routines.py according to experiment setup to avoid overwriting. 
6. Run embs_test_runner.py
