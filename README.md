# RetiZero
# Abstract
Previous foundation models for retinal images were pre-trained with limited disease categories and knowledge base. Here we introduce RetiZero, a vision-language foundation model that leverages knowledge from over 400 fundus diseases. To RetiZero’s pre-training, we compiled 341,896 fundus images paired with text descriptions, sourced from public datasets, ophthalmic literature, and online resources, encompassing a diverse range of diseases across multiple ethnicities and countries. RetiZero exhibits superior performance in several downstream tasks, including zero-shot disease recognition, image-to-image retrieval, and internal- and cross-domain disease identification. In zero-shot scenarios, RetiZero achieves Top5 accuracy scores of 0.8430 for 15 fundus diseases and 0.7561 for 52 fundus diseases. For image retrieval, it achieves Top5 scores of 0.9500 and 0.8860 for the same disease sets, respectively. Clinical evaluations show that RetiZero’s Top3 zero-shot performance surpasses the average of 19 ophthalmologists from Singapore, China and the United States. Furthermore, RetiZero significantly enhances clinicians’ accuracy in diagnosing fundus disease. These findings underscore the value of integrating the RetiZero foundation model into clinical settings, where a variety of fundus diseases are encountered.

![Overview](https://github.com/LooKing9218/RetiZero/assets/104180884/c5096814-fcc5-448b-a22e-148648e72df0)


# Create and Activate Conda Environment
conda create -n retizero python=3.8 -y

conda activate retizero

# Install Dependencies
pip install -r requirements.txt
