import numpy as np


def result_filter(results:list, class_list:list, class_threshold:list, inter_threshold:float=0.5):

    if results == []:
        return results
    
    new_results = []
    for i in range(len(class_list)):

        class_confident_scores = [result[2][i] for result in results]
        confident_score_order = sorted(range(len(class_confident_scores)), key=lambda x:class_confident_scores[x], reverse=True)
        class_results = [results[x] for x in confident_score_order]

        while len(class_results) > 0:

            bh = class_results[0] # box with highest score
            if bh[2][i] < class_threshold[i]:
                break
            label = np.array([False for _ in class_list])
            label[i] = True  
            class_results[0][2] = label
            new_results.append(class_results.pop(0))
            
            for box in class_results:

                Area_of_Intersection = \
                    max(0, min(bh[1][1] + bh[0], box[1][1] + box[0]) - max(bh[1][1], box[1][1])) * \
                    max(0, min(bh[1][0] + bh[0], box[1][0] + box[0]) - max(bh[1][0], box[1][0]))
                Area_of_Union = bh[0]**2 + box[0]**2 - Area_of_Intersection
                IoU = Area_of_Intersection / Area_of_Union

                if IoU >= inter_threshold:
                    class_results.remove(box)

    return new_results