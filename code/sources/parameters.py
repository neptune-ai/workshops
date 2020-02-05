PARAMS = {'batch_size': 16,  # 16, 32, 64
          'n_epochs': 100,
          'shuffle': True,
          'activation': 'elu',  # 'elu', 'relu'
          'dense_units': 32,  # 16, 32, 64, 128
          'dropout': 0.28,  # float between 0.0 and 1.0
          'learning_rate': 0.003,  # float between, say, 0.00001 and 0.01
          'early_stopping': 10,
          'optimizer': 'Nadam',  # 'Adam', 'Nadam', SGD'
          }
