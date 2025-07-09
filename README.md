# Abstract
Previous foundation models for fundus images were pre-trained with limited disease categories and knowledge base. Here we introduce RetiZero, a vision-language model that incorporates knowledge from over 400 fundus diseases. The model is pre-trained on 341,896 fundus images with accompanying text descriptions gathered from diverse sources across multiple ethnicities and countries. RetiZero demonstrates exceptional performance across various downstream tasks including zero-shot disease recognition, image-to-image retrieval, clinical diagnosis assistance, few-shot fine-tuning, and cross-domain disease identification. In zero-shot scenarios, it achieves Top-5 accuracies of 0.843 for 15 diseases and 0.756 for 52 diseases, while for image-to-image retrieval, it scores 0.950 and 0.886 respectively. Notably, RetiZero’s Top-3 zero-shot performance exceeds the average diagnostic accuracy of 19 ophthalmologists from Singapore, China, and the United States. The model particularly enhances clinicians’ ability to diagnose rare fundus conditions, highlighting its potential value for integration into clinical settings where diverse eye diseases are encountered.


![Overview](https://github.com/user-attachments/assets/12ef87c1-e178-4911-b3e4-86647fb2a749)





# Create and Activate Conda Environment
conda create -n retizero python=3.8 -y

conda activate retizero

# Install Dependencies
pip install -r requirements.txt

# Please download the pre-trained weights for RetiZero through the following link:

https://drive.google.com/file/d/14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM/view?usp=sharing



# Citation
@article{wang2025enhancing,
  title={Enhancing diagnostic accuracy in rare and common fundus diseases with a knowledge-rich vision-language model},
  author={Wang, Meng and Lin, Tian and Lin, Aidi and Yu, Kai and Peng, Yuanyuan and Wang, Lianyu and Chen, Cheng and Zou, Ke and Liang, Huiyu and Chen, Man and others},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={5528},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
