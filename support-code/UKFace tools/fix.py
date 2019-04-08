import os


folder = "C:/Users/foggy/facialRecognition/output/intermediate/19-25/"

for file in os.listdir(folder):
    splitby_ = file.split('_')
    splitbydash = splitby_[0].split('-')
    new_name = "19-" + splitbydash[1] + "_" + splitby_[1]
    full_new_name = folder + new_name
    print(full_new_name)
    os.rename(folder+file, full_new_name)
