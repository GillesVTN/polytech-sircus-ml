import os
import pandas as pd
import multiprocessing as mp

class CsvBuilder:

    def __init__(self):
        self.path = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"
        self.images_names = ["mos01.png", "mos02.png", "mos03.png", "mos04.png", "mosob01.jpg", "mosob02.jpg",
                             "mosob03.jpg",
                             "mosob04.jpg", "ob01.jpg", "ob02.jpg", "ob03.jpg", "ob04.jpg", "vis01.jpg", "vis02.jpg",
                             "vis03.jpg", "vis04.jpg", "vismos01.jpg", "vismos02.jpg", "vismos03.jpg", "vismos04.jpg",
                             "visob01.jpg", "visob02.jpg", "visob03.jpg", "visob04.jpg"]

    def find_all_datas(self):
        """
        Renvoie la liste des fichiers d'un certain dossier
        :param group(str): groupe, nom du dosssier où sont stockés les données
        :return: files_names(list[str])
        """

        groups = ["Controles", "Patients"]

        files_names = []

        for group in groups:
            for file in os.listdir(os.path.abspath("{}/{}".format(self.path, group))):
                files_names.append("{}/{}".format(group.__str__(), os.fsdecode(file)))

        return files_names

    def chunk_list(self, list, n):
        """

        :return:
        """
        for i in range(0, len(list), n):
            yield list[i:i+n]

    def generate_csv(self, jobs_nb=10):

        files_list = self.find_all_datas()

        jobs = []
        progress = 0

        with mp.Pool(os.cpu_count()) as pool:
            for chunk in self.chunk_list(files_list, jobs_nb):
                jobs.append(pool.apply_async(self.generate_csv_range, [chunk]))
                print("job started.")

            for job in jobs:
                job.get()
                print("Job ", jobs.index(job), "/", jobs_nb, " - Progress ", (jobs.index(job)/jobs_nb)*100, " %")

    def generate_csv_range(self, files_list):
        """
        Gènere un fichier csv receuillant toutes les données(x,y,image,leftdiam,rightdiam) par patient
        :return: None
        """

        for file_name in files_list:

            print("file {}/{}".format(files_list.index(file_name), len(files_list)))

            csv_output = pd.DataFrame(columns=["image_name", "number", "x", "y", "left_diam", "right_diam", "time"])

            path = "{}/{}".format(self.path, file_name)

            for image_name in self.images_names:
                sheet_name = "{} - G".format(image_name)
                try:
                    sheets = pd.read_excel(path, sheet_name=sheet_name, header=1)
                    for index, row in sheets.iterrows():
                        new_row = {"image_name": image_name, "number": row["Number"], "x": row["x"], "y": row["y"],
                                   "left_diam": row["Left Diam"], "right_diam": row["Right Diam"], "time": row["Time"]}
                        # print(sheet_name, new_row)
                        csv_output = csv_output.append(new_row, ignore_index=True)

                except ValueError as ve:
                    print(ve.__str__())

            # print(csv_output)

            if not csv_output.empty:
                print(file_name)
                csv_output.to_csv(path_or_buf=os.path.abspath("output//{}.csv".format(file_name.split(".xlsx")[0].split("/")[1])),
                                  index=False)

