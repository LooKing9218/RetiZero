![image](https://github.com/user-attachments/assets/6b90cd5f-57ff-419e-85cf-efa8ccb0397a)# RetiZero
# Abstract
Previous foundation models for fundus images were pre-trained with limited disease categories and knowledge base. Here we introduce a knowledge-rich vision-language model (RetiZero) that leverages knowledge from more than 400 fundus diseases. For RetiZero’s pre-training, we compiled 341,896 fundus images paired with texts, sourced from public datasets, ophthalmic literature, and online resources, encompassing a diverse range of diseases across multiple ethnicities and countries. RetiZero exhibits remarkable performance in several downstream tasks, including zero-shot disease recognition, image-to-image retrieval, AI-assisted clinical diagnosis, and internal- and cross-domain disease identification, and few-shot fine-tuning. In zero-shot scenarios, RetiZero achieves Top-5 accuracies of 0.843 for 15 diseases and 0.756 for 52 diseases. For image retrieval, it achieves Top-5 scores of 0.950 and 0.886 for the same sets, respectively. AI-assisted clinical diagnosis results show that RetiZero’s Top-3 zero-shot performance surpasses the average of 19 ophthalmologists from Singapore, China, and the United States. RetiZero substantially enhances clinicians’ accuracy in diagnosing fundus diseases, in particularly rare ones. These findings underscore the value of integrating the RetiZero into clinical settings, where various fundus diseases are encountered.


![Overview](https://github.com/user-attachments/assets/12ef87c1-e178-4911-b3e4-86647fb2a749)





# Create and Activate Conda Environment
conda create -n retizero python=3.8 -y

conda activate retizero

# Install Dependencies
pip install -r requirements.txt
