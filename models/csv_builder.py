import os
import numpy as np
import pandas as pd


class CsvBuilder:

    def __init__(self):
        self.path = "Acquisitions_Eye_tracking_objets_visages_Fix_Seq1"
        self.images_names = ["mos01.png", "mos02.png", "mos03.png", "mos04.png", "mosob01.jpg", "mosob02.jpg",
                             "mosob03.jpg",
                             "mosob04.jpg", "ob01.jpg", "ob02.jpg", "ob03.jpg", "ob04.jpg", "vis01.jpg", "vis02.jpg",
                             "vis03.jpg", "vis04.jpg", "vismos01.jpg", "vismos02.jpg", "vismos03.jpg", "vismos04.jpg",
                             "visob01.jpg", "visob02.jpg", "visob03.jpg", "visob04.jpg"]

        self.control_files_names = ["C_0001_74_F_JM.xlsx", "C_0002_121_F_SJ.xlsx", "C_0004_97_M_JG.xlsx",
                                    "C_0005_70_F_JS.xlsx", "C_0006_110_F_ND.xlsx", "C_0007_83_F_FE.xlsx",
                                    "C_0008_81_F_MM.xlsx",
                                    "C_0009_114_M_HM.xlsx",
                                    "C_0010_133_M_KS.xlsx", "C_0011_126_M_AB.xlsx", "C_0012_147_F_LL.xlsx",
                                    "C_0013_103_M_DR.xlsx", "C_0014_60_M_EM.xlsx",
                                    "C_0015_45_M_NL.xlsx", "C_0016_99_M_LL.xlsx", "C_0017_66_F_LL.xlsx",
                                    "C_0018_32_M_NL.xlsx",
                                    "C_0019_96_M_BS.xlsx",
                                    "C_0020_63_M_RS.xlsx", "C_0021_115_F_LB.xlsx", "C_0022_71_M_PB.xlsx",
                                    "C_0023_121_F_RB.xlsx",
                                    "C_0024_89_M_TB.xlsx",
                                    "C_0025_71_M_CS.xlsx", "C_0026_75_M_EC.xlsx", "C_0027_105_F_IM.xlsx",
                                    "C_0028_47_M_FC.xlsx",
                                    "C_0029_82_M_TC.xlsx",
                                    "C_0030_102_M_OL.xlsx", "C_0031_53_M_VL.xlsx", "C_0032_138_F_LR.xlsx",
                                    "C_0033_114_F_MR.xlsx", "C_0034_52_F_RL.xlsx",
                                    "C_0036_111_F_AP.xlsx", "C_0037_62_F_RJ.xlsx", "C_0038_106_M_AH.xlsx",
                                    "C_0039_61_M_MH.xlsx",
                                    "C_0040_46_F_HT.xlsx",
                                    "C_0041_73_F_LI.xlsx", "C_0042_38_M_CL.xlsx", "C_0043_116_F_ML.xlsx",
                                    "C_0044_49_M_AL.xlsx",
                                    "C_0045_111_M_AC.xlsx",
                                    "C_0046_131_F_MM.xlsx", "C_0047_91_F_LL.xlsx", "C_0048_113_M_BL.xlsx",
                                    "C_0049_44_F_LC.xlsx",
                                    "C_0050_58_F_CH.xlsx",
                                    "C_0051_61_M_MC.xlsx", "C_0052_145_M_RG.xlsx", "C_0053_75_F_MS.xlsx",
                                    "C_0054_116_M_KM.xlsx",
                                    "C_0055_44_F_LE.xlsx",
                                    "C_0056_131_F_SL.xlsx", "C_0057_77_M_MG.xlsx", "C_0058_109_F_FG.xlsx",
                                    "C_0059_114_M_AC.xlsx", "C_0060_115_M_NT.xlsx",
                                    "C_0061_55_M_GA.xlsx", "C_0062_156_F_LP.xlsx", "C_0063_118_M_MP.xlsx",
                                    "C_0064_118_F_LP.xlsx", "C_0065_54_M_MU.xlsx",
                                    "C_0066_144_F_SV.xlsx", "C_0067_122_M_PV.xlsx", "C_0068_64_M_FV.xlsx",
                                    "C_0069_133_F_IK.xlsx", "C_0070_92_F_LP.xlsx",
                                    "C_0071_66_F_PM.xlsx", "C_0072_74_M_JJ.xlsx", "C_0075_102_F_MD.xlsx",
                                    "C_0076_77_M_ED.xlsx",
                                    "C_0077_136_F_AL.xlsx",
                                    "C_0078_66_M_GP.xlsx", "C_0079_34_F_EV.xlsx", "C_0080_104_F_PC.xlsx",
                                    "C_0082_67_M_AC.xlsx",
                                    "C_0083_36_M_TC.xlsx",
                                    "C_0084_126_M_EF.xlsx", "C_0085_103_M_BM.xlsx", "C_0089_46_F_BS.xlsx",
                                    "C_0090_114_F_PE.xlsx", "C_0091_74_M_PA.xlsx",
                                    "C_0092_108_M_CN.xlsx", "C_0093_85_M_CG.xlsx", "C_0096_88_F_CA.xlsx",
                                    "C_0098_76_F_GM.xlsx",
                                    "C_0100_83_M_FL.xlsx",
                                    "C_0101_114_F_JL.xlsx", "C_0102_67_M_JL.xlsx", "C_0103_129_M_AF.xlsx",
                                    "C_0104_53_M_AS.xlsx",
                                    "C_0105_80_F_RS.xlsx"]

        self.patient_files_names = ["P_0003_63_M_NJ.xlsx", "P_0004_87_M_LK.xlsx", "P_0005_70_M_TJ.xlsx",
                                    "P_0009_87_M_LR.xlsx", "P_0010_150_M_AJ.xlsx", "P_0011_73_M_JT.xlsx",
                                    "P_0012_102_M_MP.xlsx",
                                    "P_0013_79_M_HJ.xlsx",
                                    "P_0014_142_M_EJ.xlsx", "P_0015_40_M_NK.xlsx", "P_0016_32_F_PS.xlsx",
                                    "P_0017_41_M_AA.xlsx",
                                    "P_0019_55_F_BM.xlsx",
                                    "P_0020_55_M_YS.xlsx", "P_0022_69_M_RR.xlsx", "P_0023_100_M_GM.xlsx",
                                    "P_0024_46_F_GS.xlsx",
                                    "P_0025_125_M_MC.xlsx",
                                    "P_0026_43_M_AG.xlsx", "P_0027_55_F_AY.xlsx", "P_0028_64_M_ST.xlsx",
                                    "P_0029_72_M_BK.xlsx",
                                    "P_0031_103_M_NM.xlsx",
                                    "P_0032_123_M_JA.xlsx", "P_0033_65_M_LJC.xlsx", "P_0034_65_M_JC.xlsx",
                                    "P_0035_102_M_NE.xlsx", "P_0036_112_M_DY.xlsx",
                                    "P_0037_64_M_PG.xlsx", "P_0038_123_F_LG.xlsx", "P_0039_63_M_TS.xlsx",
                                    "P_0042_136_M_EB.xlsx",
                                    "P_0042_87_M_EC.xlsx",
                                    "P_0043_51_M_JB.xlsx", "P_0044_118_M_WL.xlsx", "P_0045_118_M_VL.xlsx",
                                    "P_0046_102_M_JM.xlsx", "P_0048_144_M_JM.xlsx",
                                    "P_0049_128_M_AH.xlsx", "P_0050_113_F_MB.xlsx", "P_0051_152_M_LT.xlsx",
                                    "P_0052_151_M_EB.xlsx", "P_0053_74_M_MD.xlsx",
                                    "P_0055_129_M_GS.xlsx", "P_0057_67_M_K.xlsx", "P_0058_76_M_EF.xlsx",
                                    "P_0059_73_M_TM.xlsx",
                                    "P_0060_93_M_DK.xlsx",
                                    "P_0061_66_M_AD.xlsx", "P_0062_157_M_AF.xlsx", "P_0063_82_M__KE.xlsx",
                                    "P_0064_79_M_ES.xlsx",
                                    "P_0066_99_F_MC.xlsx",
                                    "P_0067_73_M_SB.xlsx", "P_0069_49_M_EZ.xlsx", "P_0070_134_M_LL.xlsx",
                                    "P_0071_103_M_AD.xlsx",
                                    "P_0073_69_M_TM.xlsx",
                                    "P_0074_121_M_AF.xlsx", "P_0076_60_M_TL.xlsx", "P_0077_95_M_AB.xlsx",
                                    "P_0080_113_M_MJC.xlsx", "P_0083_102_M_NL.xlsx",
                                    "P_0085_203_F_TA.xlsx", "P_0086_84_M_IN.xlsx", "P_0090_114_M_NB.xlsx",
                                    "P_0091_73_M_MP.xlsx",
                                    "P_0092_157_F_LC.xlsx",
                                    "P_0094_M_VD.xlsx", "P_0095_82_M_MD.xlsx", "P_0099_88_M_RM.xlsx",
                                    "P_0100_144_M_M.xlsx",
                                    "P_0111_100_M_N.xlsx"]
    def find_all_datas(self, group:str):
        """
        Renvoie la liste des fichiers d'un certain dossier
        :param group(str): groupe, nom du dosssier où sont stockés les données
        :return: files_names(list[str])
        """

        files_names = []
        for file in os.listdir(os.path.abspath("{}/{}".format(self.path, group))):
            files_names.append(os.fsdecode(file))

        return files_names

    def generate_csv(self):
        """
        Gènere un fichier csv receuillant toutes les données(x,y,image,leftdiam,rightdiam) par patient
        :return: None
        """
        current_group = self.control_files_names
        group_type = "Controles"

        for file_name in current_group:

            print("file {}/{}".format(current_group.index(file_name), len(current_group)))

            csv_output = pd.DataFrame(columns=["image_name", "number", "x", "y", "left_diam", "right_diam", "time"])

            path = "{}/{}/{}".format(self.path, group_type, file_name)

            for image_name in self.images_names:
                sheet_name = "{} - G".format(image_name)
                try:
                    sheets = pd.read_excel(path, sheet_name=sheet_name, header=1)
                    for index, row in sheets.iterrows():
                        new_row = {"image_name": image_name, "number": row["Number"], "x": row["x"], "y": row["y"],
                                   "left_diam": row["Left Diam"], "right_diam": row["Right Diam"], "time": row["Time"]}
                        #print(sheet_name, new_row)
                        csv_output = csv_output.append(new_row, ignore_index=True)

                except ValueError as ve:
                    print(ve.__str__())

            #print(csv_output)

            if not csv_output.empty:
                csv_output.to_csv(path_or_buf=os.path.abspath("output//{}.csv".format(file_name.split(".xlsx")[0])), index=False)

            if len(current_group) == current_group.index(file_name):
                current_group = self.patient_files_names
                group_type = "Patients"
