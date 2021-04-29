try:
    import numpy as np
    import torch
    from pathlib import Path
    import pickle
    import os
    import os.path as osp
    import sys
    import importlib
    from tqdm import tqdm
    importlib.reload(Config)
    importlib.reload(TaskClass)
    importlib.reload(Task)
except NameError: # It hasn't been imported yet
        import config.config_flags as Config
        import Task as Task
        import TaskClass as TaskClass

def generate_tasks(task_db, data_provider, tr_size, val_size, device):
    num_of_tasks = len(task_db)
    batch_size = Config.BATCH_SIZE
    tasks = []
    for task_num in tqdm(range(num_of_tasks)):
        task_model = Task.Task(str(task_num), batch_size)
        task_row = task_db[task_num]
        unique_classes = np.unique(task_row[1])
        
        for class_id in unique_classes:
            taskclass = TaskClass.TaskClass(class_id)
            task_model.add_task_class(taskclass)

            embeddings_list = [torch.from_numpy(data_provider.get_embeddings_for_image_file(image_file)).to(device) \
                                for image_file in task_row[2][class_id]]
            
            train_embeddings = embeddings_list[:tr_size]
            val_embeddings = embeddings_list[tr_size:]

            dataset_type = data_provider.get_dataset_type()
 
            taskclass.set_query_dataset(val_embeddings)
            taskclass.set_support_dataset(train_embeddings)

        task_model.reset_train_session()
        tasks.append(task_model)
    return tasks