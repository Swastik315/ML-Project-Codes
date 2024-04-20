from ExoCtk import ExoCtk
from FileHelper import FileHelper

exo_ctk = ExoCtk()

def transform(data_file,temp, g,r_planet,r_star,metallicity, c_o,haze,cloud):
    res=[]
    temp_list=data_file.strip().split('\n')
    for i in temp_list:
        res.append(i.split()[1])

    res+=[temp, g,r_planet,r_star,metallicity, c_o,haze,cloud]
    return ' '.join(res)

def generateData(filename,temp, g, r_planet, r_star, metallicity, c_o, haze, cloud):
    data_file = exo_ctk.generate_values(
        temp, g, r_planet, r_star, metallicity, c_o, haze, cloud
    )
    if data_file:
        vector_file = transform(data_file,temp, g,r_planet,r_star,metallicity, c_o,haze,cloud)
        folder_path = "data_values"
        FileHelper().insertOne(folder_path, filename, vector_file)
        print("{} generated in {}".format(filename,folder_path))

choice=input("Do you want to delete prev generated data ? (y/n): ")
if(choice=="y"):
    FileHelper().removeAll("data_values")
    print("Deleted all previously generated data")

temp = ["400"]
for i in range(6, 27):
    temp.append(str(i * 100))
g = ["5", "10", "20", "50"]
r_planet = ["1","1.5"]
r_star = ["0.1","0.9"]
metallicity = ["1", "10", "50", "100", "200"]
c_o = ["0.35", "0.56", "0.70", "1.00"]
haze = ["0001", "0010", "0100", "1100"]
cloud = ["0.00", "0.06", "0.20", "1.00"]

for i in temp:
    for j in g:
        for k in r_planet:
            for l in r_star:
                for m in metallicity:
                    for n in c_o:
                        for o in haze:
                            for p in cloud:
                                filename = "{}x{}x{}x{}x{}x{}x{}x{}.txt".format(i,j,k,l,m,n,o,p)
                                if(FileHelper().findFile("data_values", filename)):
                                    print("{} already exists".format(filename))
                                    continue
                                generateData(filename,i, j, k, l, m, n, o, p)
