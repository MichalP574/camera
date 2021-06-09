import numpy as np
import math

centre = [10, 5]
perpendicular = [10, 10]
vector1 = [5, 5]
centre = np.array(centre)
perpendicular = np.array(perpendicular)
vector1 = np.array(vector1)
perpendicular_LCS = perpendicular - centre
vector_LCS = vector1 - centre
perpendicular_LCS[1] = -perpendicular_LCS[1]
vector_LCS[1] = -vector_LCS[1]
print(f"Vector:{vector_LCS}")
print(f"Perpendicular: {perpendicular_LCS}")

angle2 = math.atan2(vector_LCS[1],vector_LCS[0])
perpendicular = math.atan2(perpendicular_LCS[1],perpendicular_LCS[0])

angle2 = math.degrees(angle2)
perpendicular = math.degrees(perpendicular)
print(perpendicular, angle2)
alpha = angle2-perpendicular
print(alpha)
