import os
import pickle
from datetime import datetime
from pathlib import Path


#GPT-generated unique file name finder
def get_unique_filename(file_path):

    file_path = Path(file_path)
    base = file_path.stem  # Filename without extension
    ext = file_path.suffix  # File extension (e.g., '.txt')
    directory = file_path.parent  # Parent directory

    counter = 1
    unique_path = file_path

    # Check for uniqueness
    while unique_path.exists():
        unique_path = directory / f"{base}_{counter}{ext}"
        counter += 1

    return str(unique_path)

#saves the model in a directory dependent on the day
def save_model(model, results = None, model_name="NEAT_model.pkl", results_name="results.pkl", base_directory="Neat/models"):
    #create a new folder for the current date
    current_date = datetime.now()
    directory = os.path.join(base_directory, current_date.strftime("%m-%d-%y"))
    print("directory name:", directory)
    
    #make new dir if it doesn't exist
    if not os.path.exists(directory):
        print("making directory")
        os.makedirs(directory)
    
    model_path = os.path.join(directory, model_name)
    results_path = os.path.join(directory, results_name)

    #do not overwrite existing models!!!
    model_path = get_unique_filename(model_path)
    results_path = get_unique_filename(results_path)

    #save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    #save fitness data
    with open(results_path, "wb") as f:
        if results is not None:
            pickle.dump(results, f)

def load_latest(directory = "NEAT/models", model_name="NEAT_model.pkl", results_name=""):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    #parse folder names for dates
    valid_dates = []
    for folder in folders:
        try:
            parsed_date = datetime.strptime(folder, "%m-%d-%y")
            valid_dates.append((parsed_date, folder))
        except ValueError:
            #skip folders that don't match the date format
            pass

    #no folders exist
    if not valid_dates:
        print("No models found.")
        return None

    #select and return the latest model
    _, latest_folder = max(valid_dates)
    folder_path = os.path.join(directory, latest_folder)
    print(f"loading model from {os.path.join(folder_path, model_name)}")

    #load model
    with open(os.path.join(folder_path, model_name), 'rb') as f:
        model = pickle.load(f)

    if results_name == "": #no results data requested
        return model, []
    
    #load results data
    with open(os.path.join(folder_path, results_name), 'rb') as f:
        results = pickle.load(f)
    
    return model, results
