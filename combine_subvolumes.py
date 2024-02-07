import shutil
import os
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt

class comboFolders:
    def __init__(self, parent, combo, overlap, ref_overlap):
        self.parent = parent
        self.combo = combo
        self.ref_overlap = ref_overlap
        self.overlap = overlap
        self.sections = os.listdir(parent)
        self.section_paths = [os.path.join(self.parent, i).replace("\\", "/") for i in self.sections]
        self.section_img_paths = [self.sortPaths(os.listdir(i)) for i in self.section_paths]
        self.matches = self.matchDict()
        self.detOverlap()
        self.writeComboImages()
        print('hello')

    def matchDict(self):
        matches = dict()
        for index, i in enumerate(self.sections):
            matches[str(index)] = dict()
            matches[str(index)]['start'] = 0
            matches[str(index)]['end'] = len(self.section_img_paths[index])
        return matches

    def makeDir(self, parent, new_path):
        try:
            os.mkdir(os.path.join(parent, new_path).replace("\\","/"))
        except OSError as error:
            print(error)

    def sortPaths(self, paths):
        paths = [i for i in paths if i.endswith(".tif") or i.endswith(".tiff")]
        sort_paths = sorted(paths, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        return sort_paths
    def pathComb(self, parent, sub):
        path = os.path.join(parent, sub).replace("\\", "/")
        return path
    def binaryOnly(self, img):
        img[img < 255] = 0
        img[img > 0] = 1
        return img

    def iouScore(self, img1_path, img2_path):
        img1 = tif.imread(img1_path)
        img2 = tif.imread(img2_path)
        img1 = self.binaryOnly(img1)
        img2 = self.binaryOnly(img2)
        intersection = img1*img2 #logical and
        union = img1 + img2 #logical or
        union[union > 1] = 1
        IOU = intersection.sum()/float(union.sum())
        return IOU

    def detOverlap(self):
        for i in range(1, len(self.sections)):
            self.bestMatch(i-1, i)


    def bestMatch(self, sec1_index, sec2_index):
        sec1 = self.section_img_paths[sec1_index]
        sec2 = self.section_img_paths[sec2_index]

        # COPIES EACH IMAGE IN SAMPLE SECTION FOLDER TO COMBINED FOLDER
        # LIST OF FILE LOCATIONS FOR EACH IMAGE IN THE SAMPLE COMBINED

        sec1_len = len(sec1)
        # FILE NAMES FOR BEST MATCH FROM EACH REFERENCE IMAGE
        best_matches = []
        # CORRESPONDING IOU SCORES FOR BEST_MATCHES IMAGES
        best_scores = []

        for r in range(self.ref_overlap):
            # STORAGE FOR IMAGE SIMILARITY (IOU) SCORES
            sim_list = list()
            img2_path = self.pathComb(self.section_paths[sec2_index], sec2[r])
            decline = 0
            # o_lap NUMBER OF IMAGES IS SCREENED FOR DUPLICATES
            for i in range(sec1_len - self.overlap, sec1_len):
                img1_path = self.pathComb(self.section_paths[sec1_index], sec1[i])
                score = self.iouScore(img1_path, img2_path)
                if i == sec1_len - self.overlap:
                    Temp = score
                if score < Temp:
                    decline += 1
                if score > Temp:
                    decline = 0
                # DECLINE USED TO EXIT FOR LOOP IF
                # IOU SCORE DOES NOT IMPROVE
                if decline == 25:
                    break
                # SIMILARITY SCORE IS DETERMINED AND STORED IN SIM_LIST
                sim_list.append(score)
                Temp = max(sim_list)
            # MAXIMUM VALUE OF THE SIM_LIST IS DETERMINED
            max_sim = max(sim_list)
            # MOST SIMILAR IMAGE LOCATION IS DETERMINED
            index_max_sim = sim_list.index(max_sim)
            best_scores.append(max_sim)
            best_matches.append(index_max_sim + sec1_len - self.overlap)

        best_scores = np.array(best_scores)
        self.matches[str(sec1_index)]['end'] = best_matches[np.argmax(best_scores)]
        self.matches[str(sec2_index)]['start'] = np.argmax(best_scores)

    def writeComboImages(self):
        count = 0
        for index, sec in enumerate(self.sections):
            for img in range(self.matches[str(index)]['start'], self.matches[str(index)]['end']):
                OG = self.pathComb(self.section_paths[index], self.section_img_paths[index][img])
                shutil.copy(OG, self.combo)
                RENAMED = self.pathComb(self.combo, str(count)+'_'+self.section_paths[index].split('/')[-1]+'_'+self.section_img_paths[index][img])
                comb_OG = self.pathComb(self.combo, self.section_img_paths[index][img])
                os.rename(comb_OG, RENAMED)
                count += 1

parent = 'M:/mcgrath/data_cleaning/cleaned_img/sample_1/segmented_cleaned'
overlap = 200
ref_overlap = 3
combine_path = 'C:/testtest'
test = comboFolders(parent, combine_path, overlap, ref_overlap)