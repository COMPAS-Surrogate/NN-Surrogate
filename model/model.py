from .base_model import BaseModel
from dataloader.dataloader import DataLoader
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
tfpl = tfp.layers
tfpd = tfp.distributions

device_type = 'GPU'
n_gpus = 2
devices = tf.config.experimental.list_physical_devices(
          device_type)
devices_names = [d.name.split('e:')[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(
          devices=devices_names[:n_gpus])


class COMPAS_NN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_train = self.config.train.num_training_samples
        self.test_samples_frac = self.config.train.test_samples_frac
        self.validation_samples_frac = self.config.train.validation_samples_frac
        self.dataset = self.config.train.dataset
        self.batch_size = self.config.train.batch_size
        self.epochs = self.config.train.epoches
        self.checkpoint_path = self.config.train.checkpoint_path
        self.model_path = self.config.train.model_path
        self.train_from_checkpoint = self.config.train.train_from_checkpoint
        self.lr = self.config.model.learning_rate

        self.feature_scaler = StandardScaler()
        self.label_scaler = StandardScaler()
        
        self.checkpoint_path = self.config.train.checkpoint_path

    def load_and_preprocess_data(self):
        
        self.samples, self.lnl_values = DataLoader(self.dataset).load_data(self.config.data)

        # Filter out -inf values
        valid_indices = ~np.isinf(self.lnl_values)
        self.samples_filtered = self.samples[valid_indices]
        self.lnl_values_filtered = self.lnl_values[valid_indices]

        # Normalize the features and labels
        self.samples_scaled = self.feature_scaler.fit_transform(self.samples_filtered)
        self.lnl_values_scaled = self.label_scaler.fit_transform(self.lnl_values_filtered.reshape(-1, 1)).squeeze()

        # Split the data into training, validation, and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.samples_scaled, self.lnl_values_scaled, test_size=self.test_samples_frac, random_state=0)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=self.validation_samples_frac, random_state=0)

        # Convert to float32
        self.y_train = self.y_train.astype(np.float32)
        self.y_test = self.y_test.astype(np.float32)
        self.y_val = self.y_val.astype(np.float32)

        
    def build(self):
        with strategy.scope():
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.samples.shape[1],)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(1)),
                tfp.layers.IndependentNormal(1)
            ])
            negloglik = lambda y, rv_y: -rv_y.log_prob(y)
            optimizer = tf.keras.optimizers.Adam(lr=self.lr)
            self.model.compile(optimizer=optimizer, loss=negloglik)
            self.model.summary()
        
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self.model)

    def plot_loss_curves(self, loss, val_loss):
    
        # summarize history for accuracy and loss
        plt.figure(figsize=(6, 4))
        plt.plot(loss, "r--", label="Loss on training data")
        plt.plot(val_loss, "r", label="Loss on validation data")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.savefig("/fred/oz016/Chayan/COMPAS_populations_project/NN-Surrogate/evaluation/Loss_curve_"+str(self.dataset)+"_data.png", dpi=200)
    

    def train(self, checkpoint):
        """Trains the model"""
        with strategy.scope():
            # initialize checkpoints
            dataset_name = self.config.train.checkpoint_path
            checkpoint_directory = "{}/tmp_{}".format(dataset_name, str(hex(random.getrandbits(32))))
            checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

            # load best model with min validation loss
            if(self.train_from_checkpoint == True):
                checkpoint.restore(self.checkpoint_path)

            model_history = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                              validation_data=(self.X_val, self.y_val), verbose=1)
        
            checkpoint.save(file_prefix=checkpoint_prefix)
        
            self.model.save(self.model_path)
            self.plot_loss_curves(model_history.history['loss'], model_history.history['val_loss'])
        
    
    def evaluate(self):

        yhat = self.model(self.X_test)
        mean_preds_scaled = yhat.mean().numpy().squeeze()
        std_preds_scaled = yhat.stddev().numpy().squeeze()
        self.mean_preds = self.label_scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).squeeze()
        self.std_preds = std_preds_scaled
        self.y_test_rescaled = self.label_scaler.inverse_transform(self.y_test.reshape(-1, 1)).squeeze()
                  

    def plot_results(self):

        plt.errorbar(self.y_test_rescaled, self.mean_preds, yerr=self.std_preds, fmt='o', markersize=2, alpha=0.5)
        plt.plot(self.y_test_rescaled, self.y_test_rescaled, 'r--')  # y=x line for reference
        plt.xlabel('True LnL')
        plt.ylabel('Predicted LnL')
        plt.title('Predicted vs True LnL with Uncertainty')
        plt.savefig('/fred/oz016/Chayan/COMPAS_populations_project/NN-Surrogate/evaluation/Test_NN_surrogate_4_params_uncertainty_'+str(self.dataset)+'_data.png', dpi=400)


