# BOOSTING IMPERCEPTIBILITY OF ADVERSARIAL ATTACKS FOR ENVIRONMENTAL SOUND CLASSIFICATION

This is the code submitted for the anonymously submitted paper "Boosting Imperceptibility of Adversarial Attacks for
Environmental Sound Classification" for ICME2024.

<div align="center">
  <img src="framework.png" width="1000px" />
  <p>Framework of our FreqPsy Attack Algorithm</p>
</div>

## Requirements

````
numpy==1.26.2
librosa==0.10.1
pystoi==0.3.3
pandas==1.5.3
sklearn==0.0.post1
scikit-learn==1.1.3
eagerpy==0.30.0
foolbox==3.3.3
tqdm==4.64.1
scipy==1.11.4
````

## Training Model

Train two models including VGG13 and VGG16. Please download the dataset from the following link and put it in the
`dataset` folder. 

[UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)

[ESC-50](https://github.com/karolpiczak/ESC-50)

Below is an example.
```
cd FreqPsyAttack
python train_model.py --dataset Urban8K --model VGG13 --batch_size 32 --num_epochs 100 --lr 0.0001
```

## Select a portion of the data that each model can predict successfully

We have already completed this step for you and saved it under the `urban8K_data` and `esc_data`. Specifically, we
filter 50 data per class in the UrbanSound8K dataset, and select 10 data per class in the ESC-50 dataset.

## Generate adversarial examples

We use four attack algorithms to generate adversarial examples, including FGSM, BIM, PGD, and our proposed FPAA. The
following is an example. If you run the method we proposed, you need to specify the values of lr_stage, num_iter_stage,
and alpha. The other methods do not need to be specifically specified (because these parameters are the parameters used
in the method we proposed).

```
python run_attack.py --dataset Urban8K --model VGG13 --lr_stage 0.001 --num_iter_stage 500 --epsilon 0.1 --attack_method PGD_freq --save_path adv_example --alpha 0.07
```
We have generated adversarial examples on the ESC-50 data set.