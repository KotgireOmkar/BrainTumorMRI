import os
import h5py
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'model', 'best_mri_classifier.h5')

with h5py.File(MODEL_PATH, 'r+') as f:
    config = f.attrs.get('model_config')
    config = config.decode('utf-8') if isinstance(config, bytes) else config
    d = json.loads(config)
    d_str = json.dumps(d).replace('"batch_shape"', '"batch_input_shape"')
    f.attrs.modify('model_config', d_str.encode('utf-8'))
    print('Patched!')
