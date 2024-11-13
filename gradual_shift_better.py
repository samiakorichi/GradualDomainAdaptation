import utils
import models
import datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import lightgbm as lgb
import logging
import random

#logging.getLogger("lightgbm").setLevel(logging.ERROR)

def new_model_simple():
    # model = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=1000)  # max_iter pour garantir la convergence
    # model = RandomForestClassifier(
    #     n_estimators=100,  
    #     max_depth=50,  
    #     min_samples_split=5, 
    #     min_samples_leaf=2,  
    #     bootstrap=True,  # Utiliser le bootstrap pour améliorer la stabilité du modèle
    #     random_state=42  
    # )
    model = GaussianNB()
    # model = SVC(kernel='linear', probability=True, random_state=42, C=0.1, class_weight='balanced')
    # model = SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma=0.1, class_weight='balanced')
    # model = LDA()
    # model = lgb.LGBMClassifier(
    # boosting_type='gbdt',  # Arbre de décision à gradient boosté
    # num_leaves=31,  # Nombre de feuilles
    # learning_rate=0.1,  # Taux d'apprentissage
    # n_estimators=100,  # Nombre d'arbres
    # max_depth=-1,  # Profondeur illimitée de l'arbre
    # random_state=42,
    # force_col_wise=True,
    # verbosity=-1
    # )
    #model = DecisionTreeClassifier(max_depth=50, splitter='random')
    return model

def run_experiment_simple(
    dataset_func, n_classes, input_shape, save_file, model_func=new_model_simple,
    interval=2000, soft=False, conf_q=0.1, num_runs=20, num_repeats=None):

    # Charger les données prétraitées
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()

    if num_repeats is None:
        num_repeats = int(inter_x.shape[0] / interval)

    def student_func(teacher):
        return teacher

    def run(seed):
              
        utils.rand_seed(seed)
        trg_eval_x = trg_val_x
        trg_eval_y = trg_val_y

        # Entraîner le modèle source.
        source_model = new_model_simple()
        source_model.fit(src_tr_x, src_tr_y)
        src_acc = source_model.score(src_val_x, src_val_y)
        target_acc = source_model.score(trg_eval_x, trg_eval_y)
        print(f"Précision de validation source (seed {seed}): {src_acc * 100:.2f}%")
        print(f"Précision de validation cible (seed {seed}): {target_acc * 100:.2f}%")

        # Auto-entrainement graduel avec enregistrement des confiances
        print("\n\n Auto-entrainement graduel:")
        teacher = new_model_simple()
        teacher.fit(src_tr_x, src_tr_y)
        gradual_accuracies, student = utils.gradual_self_train_simple(
                             
            student_func, teacher, inter_x, inter_y, interval, soft=soft,
            confidence_q=conf_q)
            
    # Calculer la confiance moyenne pour chaque étape d'auto-entraînement
        gradual_confidences = []
        for i, x_batch in enumerate(inter_x.reshape(-1, interval, *inter_x.shape[1:])):
                                  
            probas = student.predict_proba(x_batch)  # Obtenir les probabilités de chaque classe
            max_probas = probas.max(axis=1)  # Prendre la probabilité maximale pour chaque échantillon
            mean_confidence = max_probas.mean()  # Calculer la confiance moyenne
            gradual_confidences.append(mean_confidence)              
            acc = student.score(trg_eval_x, trg_eval_y)
            gradual_accuracies.append(acc)
            gradual_confidences.append(mean_confidence)  # Ajouter la dernière étape de confiance
            

        for i, (acc, conf) in enumerate(zip(gradual_accuracies, gradual_confidences)):
                                   
            print(f"Précision après étape {i+1}: {acc * 100:.2f}% | Confiance moyenne: {conf * 100:.2f}%")

            # Bootstrap direct vers la cible
            print("\n\n Bootstrap direct vers la cible:")
            teacher = new_model_simple()
            teacher.fit(src_tr_x, src_tr_y)
            target_accuracies, _ = utils.self_train_simple(
                
                
                student_func, teacher, dir_inter_x, target_x=trg_eval_x,
                target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)

    
          # Calcul de la confiance pour le bootstrap direct
            target_confidences = []
            for i, x_batch in enumerate(dir_inter_x.reshape(-1, interval, *dir_inter_x.shape[1:])):
                            
                probas = teacher.predict_proba(x_batch)
                max_probas = probas.max(axis=1)
                target_confidences.append(max_probas.mean())
              
    
            for i, (acc, conf) in enumerate(zip(target_accuracies, target_confidences)):
                           
                print(f"Précision après étape {i+1}: {acc * 100:.2f}% | Confiance moyenne: {conf * 100:.2f}%")
                # Bootstrap direct vers toutes les données non supervisées
                print("\n\n Bootstrap direct vers toutes les données non supervisées:")
                teacher = new_model_simple()
                teacher.fit(src_tr_x, src_tr_y)
                all_accuracies, _ = utils.self_train_simple(
                  
                                    
                  student_func, teacher, inter_x, target_x=trg_eval_x,
                  target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
    
               # Calcul de la confiance pour toutes les données non supervisées
               all_confidences = []
               for i, x_batch in enumerate(inter_x.reshape(-1, interval, *inter_x.shape[1:])):
                 
                   probas = teacher.predict_proba(x_batch)
                   max_probas = probas.max(axis=1)
                   all_confidences.append(max_probas.mean())
    
               for i, (acc, conf) in enumerate(zip(all_accuracies, all_confidences)):
                 
                 
                   print(f"Précision après étape {i+1}: {acc * 100:.2f}% | Confiance moyenne: {conf * 100:.2f}%")

    return src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies, gradual_confidences, target_confidences, all_confidences


    results = []
    for i in range(num_runs):
        results.append(run(i))
    print('Enregistrement dans ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


# Compilation du modèle de réseau de neurones
def compile_model(model, loss='ce'):
    loss = models.get_loss(loss, model.output_shape[1]) # La forme de sortie du modèle model.output_shape est un tuple, sous la forme (batch_size, num_classes)
    model.compile(optimizer='adam',
                  loss=[loss],
                  metrics=[metrics.sparse_categorical_accuracy])

# Entraîner le modèle sur le jeu de données source
def train_model_source(model, split_data, epochs=1000): # epochs, nombre d'itérations d'entraînement
    model.fit(split_data.src_train_x, split_data.src_train_y, epochs=epochs, verbose=False)
    print("Précision sur les données source:")
    _, src_acc = model.evaluate(split_data.src_val_x, split_data.src_val_y)
    print("Précision sur les données cible:")
    _, target_acc = model.evaluate(split_data.target_val_x, split_data.target_val_y)
    return src_acc, target_acc

# interval : nombre d'exemples de données à utiliser à chaque étape ; conf_q : paramètre de sélection de confiance
# Le nombre d'itérations (epochs) correspond au nombre de fois que le modèle s'entraîne sur l'ensemble des données d'entraînement
# inter_x : utilisé pour l'auto-entrainement graduel ; dir_inter_x : utilisé pour l'auto-entrainement direct
def run_experiment(
    dataset_func, n_classes, input_shape, save_file, model_func=models.simple_softmax_conv_model,
    interval=2000, epochs=10, loss='ce', soft=False, conf_q=0.1, num_runs=20, num_repeats=None):
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = dataset_func()
    if soft: # Mettre à 1 la position de la classe la plus probable
        src_tr_y = to_categorical(src_tr_y)
        src_val_y = to_categorical(src_val_y)
        trg_eval_y = to_categorical(trg_eval_y)
        dir_inter_y = to_categorical(dir_inter_y)
        inter_y = to_categorical(inter_y)
        trg_test_y = to_categorical(trg_test_y)
    if num_repeats is None:
        num_repeats = int(inter_x.shape[0] / interval)
    def new_model():
        model = model_func(n_classes, input_shape=input_shape)
        compile_model(model, loss)
        return model
     # Le modèle enseignant est le modèle initial utilisé dans le processus d'auto-entrainement
    # Le modèle étudiant apprend à partir des pseudo-étiquettes générées par le modèle enseignant
    def student_func(teacher):
        return teacher
    def run(seed):
        utils.rand_seed(seed)
        trg_eval_x = trg_val_x
        trg_eval_y = trg_val_y
        # Train source model.
        source_model = new_model()
        source_model.fit(src_tr_x, src_tr_y, epochs=epochs, verbose=False)
        _, src_acc = source_model.evaluate(src_val_x, src_val_y)
        _, target_acc = source_model.evaluate(trg_eval_x, trg_eval_y)
        # Gradual self-training.
        print("\n\n Gradual self-training:")
        teacher = new_model()
        teacher.set_weights(source_model.get_weights()) # Donner les poids (paramètres) du source_model au teacher_model
        gradual_accuracies, student = utils.gradual_self_train(
            student_func, teacher, inter_x, inter_y, interval, epochs=epochs, soft=soft,
            confidence_q=conf_q)
        _, acc = student.evaluate(trg_eval_x, trg_eval_y)
        gradual_accuracies.append(acc)
        # Train to target.
        print("\n\n Direct boostrap to target:")   # Introduire directement les données non supervisées du domaine cible
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        target_accuracies, _ = utils.self_train(
            student_func, teacher, dir_inter_x, epochs=epochs, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        print("\n\n Direct boostrap to all unsup data:")  # Utiliser directement toutes les données non supervisées du domaine intermédiaire
        teacher = new_model()
        teacher.set_weights(source_model.get_weights())
        all_accuracies, _ = utils.self_train(
            student_func, teacher, inter_x, epochs=epochs, target_x=trg_eval_x,
            target_y=trg_eval_y, repeats=num_repeats, soft=soft, confidence_q=conf_q)
        return src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies
    results = []
    for i in range(num_runs):
        results.append(run(i))
    print('Saving to ' + save_file)
    pickle.dump(results, open(save_file, "wb"))


def experiment_results(save_name):
    results = pickle.load(open(save_name, "rb"))
    src_accs, target_accs = [], []
    final_graduals, final_targets, final_alls = [], [], []
    best_targets, best_alls = [], []
    for src_acc, target_acc, gradual_accuracies, target_accuracies, all_accuracies in results:
        src_accs.append(100 * src_acc)
        target_accs.append(100 * target_acc)
        final_graduals.append(100 * gradual_accuracies[-1])
        final_targets.append(100 * target_accuracies[-1])
        final_alls.append(100 * all_accuracies[-1])
        best_targets.append(100 * np.max(target_accuracies))
        best_alls.append(100 * np.max(all_accuracies))
    num_runs = len(src_accs)
    mult = 1.645  # For 90% confidence intervals
    print("\nNon-adaptive accuracy on source (%): ", np.mean(src_accs),
          mult * np.std(src_accs) / np.sqrt(num_runs))
    print("Non-adaptive accuracy on target (%): ", np.mean(target_accs),
          mult * np.std(target_accs) / np.sqrt(num_runs))
    print("Gradual self-train accuracy (%): ", np.mean(final_graduals),
          mult * np.std(final_graduals) / np.sqrt(num_runs))
    print("Target self-train accuracy (%): ", np.mean(final_targets),
          mult * np.std(final_targets) / np.sqrt(num_runs))
    print("All self-train accuracy (%): ", np.mean(final_alls),
          mult * np.std(final_alls) / np.sqrt(num_runs))
    print("Best of Target self-train accuracies (%): ", np.mean(best_targets),
          mult * np.std(best_targets) / np.sqrt(num_runs))
    print("Best of All self-train accuracies (%): ", np.mean(best_alls),
          mult * np.std(best_alls) / np.sqrt(num_runs))
    



def rotated_mnist_60_conv_experiment():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)
    
def rotated_mnist_60_conv_experiment_simple():
    run_experiment_simple(
        dataset_func=datasets.rotated_mnist_60_data_func_simple,  # Fonction de dataset
        n_classes=10,  # Nombre de classes
        input_shape=None,  # Le modèle simple ne nécessite pas de spécifier la forme d'entrée
        save_file='saved_files/rot_mnist_60_conv.dat',  # Fichier pour sauvegarder les résultats
        model_func=new_model_simple,  # Utilisation de modèles de classification simples comme la régression logistique
        interval=2000,  # Intervalle de lot de données
        soft=False,  # Étiquettes dures
        conf_q=0.1,  # Seuil de confiance
        num_runs=5  # Nombre d'exécutions
    )


def portraits_conv_experiment():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def gaussian_linear_experiment():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=100, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


# Ablations below.

def rotated_mnist_60_conv_experiment_noconf():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_noconf.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def portraits_conv_experiment_noconf():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_noconf.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def gaussian_linear_experiment_noconf():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_noconf.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=100, loss='ce',
        soft=False, conf_q=0.0, num_runs=5)


def portraits_64_conv_experiment():
    run_experiment(
        dataset_func=datasets.portraits_64_data_func, n_classes=2, input_shape=(64, 64, 1),
        save_file='saved_files/portraits_64.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


#使用了一种叫做“dialing ratios”（比例调节）的数据生成方式
def dialing_ratios_mnist_experiment():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_dialing_ratios_data_func,
        n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/dialing_rot_mnist_60_conv.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def portraits_conv_experiment_more():
    run_experiment(
        dataset_func=datasets.portraits_data_func_more, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_more.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def rotated_mnist_60_conv_experiment_smaller_interval():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_smaller_interval.dat',
        model_func=models.simple_softmax_conv_model, interval=1000, epochs=10, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)


def portraits_conv_experiment_smaller_interval():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_smaller_interval.dat',
        model_func=models.simple_softmax_conv_model, interval=1000, epochs=20, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)


def gaussian_linear_experiment_smaller_interval():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_smaller_interval.dat',
        model_func=models.linear_softmax_model, interval=250, epochs=100, loss='ce',
        soft=False, conf_q=0.1, num_runs=5, num_repeats=7)



def rotated_mnist_60_conv_experiment_more_epochs():
    run_experiment(
        dataset_func=datasets.rotated_mnist_60_data_func, n_classes=10, input_shape=(28, 28, 1),
        save_file='saved_files/rot_mnist_60_conv_more_epochs.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=15, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def portraits_conv_experiment_more_epochs():
    run_experiment(
        dataset_func=datasets.portraits_data_func, n_classes=2, input_shape=(32, 32, 1),
        save_file='saved_files/portraits_more_epochs.dat',
        model_func=models.simple_softmax_conv_model, interval=2000, epochs=30, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


def gaussian_linear_experiment_more_epochs():
    d = 100        
    run_experiment(
        dataset_func=lambda: datasets.gaussian_data_func(d), n_classes=2, input_shape=(d,),
        save_file='saved_files/gaussian_more_epochs.dat',
        model_func=models.linear_softmax_model, interval=500, epochs=150, loss='ce',
        soft=False, conf_q=0.1, num_runs=5)


#Utilisé pour exécuter une série d'expériences prédéfinies et enregistrer les résultats des expériences dans un fichier spécifié
if __name__ == "__main__":
    # Main paper experiments.
    portraits_conv_experiment()
    print("Portraits conv experiment")
    experiment_results('saved_files/portraits.dat')

    rotated_mnist_60_conv_experiment()
    print("Rot MNIST conv experiment")
    experiment_results('saved_files/rot_mnist_60_conv.dat')

    gaussian_linear_experiment()
    print("Gaussian linear experiment")
    experiment_results('saved_files/gaussian.dat')

    print("Dialing MNIST ratios conv experiment")
    dialing_ratios_mnist_experiment()
    experiment_results('saved_files/dialing_rot_mnist_60_conv.dat')

    rotated_mnist_60_conv_experiment_simple()
    print("Rot MNIST conv simple experiment")

    # Without confidence thresholding.
    portraits_conv_experiment_noconf()
    print("Portraits conv experiment no confidence thresholding")
    experiment_results('saved_files/portraits_noconf.dat')
    rotated_mnist_60_conv_experiment_noconf()
    print("Rot MNIST conv experiment no confidence thresholding")
    experiment_results('saved_files/rot_mnist_60_conv_noconf.dat')
    gaussian_linear_experiment_noconf()
    print("Gaussian linear experiment no confidence thresholding")
    experiment_results('saved_files/gaussian_noconf.dat')

    # Try predicting for next set of data points on portraits.
    portraits_conv_experiment_more()
    print("Portraits next datapoints conv experiment")
    experiment_results('saved_files/portraits_more.dat')

    # Try smaller window sizes.
    portraits_conv_experiment_smaller_interval()
    print("Portraits conv experiment smaller window")
    experiment_results('saved_files/portraits_smaller_interval.dat')
    rotated_mnist_60_conv_experiment_smaller_interval()
    print("Rot MNIST conv experiment smaller window")
    experiment_results('saved_files/rot_mnist_60_conv_smaller_interval.dat')
    gaussian_linear_experiment_smaller_interval()
    print("Gaussian linear experiment smaller window")
    experiment_results('saved_files/gaussian_smaller_interval.dat')

    # Try training more epochs.
    portraits_conv_experiment_more_epochs()
    print("Portraits conv experiment train longer")
    experiment_results('saved_files/portraits_more_epochs.dat')
    rotated_mnist_60_conv_experiment_more_epochs()
    print("Rot MNIST conv experiment train longer")
    experiment_results('saved_files/rot_mnist_60_conv_more_epochs.dat')
    gaussian_linear_experiment_more_epochs()
    print("Gaussian linear experiment train longer")
    experiment_results('saved_files/gaussian_more_epochs.dat')
