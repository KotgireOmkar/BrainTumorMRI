import os
import h5py
import json

def clean_config(obj):
    if isinstance(obj, dict):
        # 1. Clean dtype if it's a DTypePolicy dictionary
        if 'dtype' in obj and isinstance(obj['dtype'], dict):
            if obj['dtype'].get('class_name') == 'DTypePolicy':
                policy_name = obj['dtype'].get('config', {}).get('name', 'float32')
                obj['dtype'] = policy_name
                print(f"Replaced DTypePolicy with '{policy_name}'")
        
        # 2. Clean up augmentation layer configurations
        if obj.get('class_name') in ['RandomFlip', 'RandomRotation', 'RandomZoom', 'RandomHeight', 'RandomWidth', 'RandomContrast', 'RandomBrightness'] and 'config' in obj:
            cfg = obj['config']
            if 'data_format' in cfg:
                del cfg['data_format']
                print(f"Removed data_format from {obj['class_name']}")
            if 'value_range' in cfg:
                del cfg['value_range']
                print(f"Removed value_range from {obj['class_name']}")
        
        # 3. Fix input_layers and output_layers format (convert flat list to list of lists)
        if 'input_layers' in obj and isinstance(obj['input_layers'], list):
            # Check if it's a flat list (e.g. first item is a string, not a list)
            if len(obj['input_layers']) > 0 and isinstance(obj['input_layers'][0], str):
                obj['input_layers'] = [obj['input_layers']]
                print(f"Nested input_layers: {obj['input_layers']}")
                
        if 'output_layers' in obj and isinstance(obj['output_layers'], list):
            if len(obj['output_layers']) > 0 and isinstance(obj['output_layers'][0], str):
                obj['output_layers'] = [obj['output_layers']]
                print(f"Nested output_layers: {obj['output_layers']}")

        # Recursively clean
        for k, v in list(obj.items()):
            clean_config(v)
    elif isinstance(obj, list):
        for item in obj:
            clean_config(item)

# Re-read and apply all rules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'model', 'best_mri_classifier.h5')

with h5py.File(MODEL_PATH, 'r+') as f:
    config = f.attrs.get('model_config')
    config = config.decode('utf-8') if isinstance(config, bytes) else config
    d = json.loads(config)
    
    clean_config(d)
    
    d_str = json.dumps(d).replace('"batch_shape"', '"batch_input_shape"')
    f.attrs.modify('model_config', d_str.encode('utf-8'))
    print('Patched DTypePolicy, data_format, value_range, and input/output layer nesting!')
