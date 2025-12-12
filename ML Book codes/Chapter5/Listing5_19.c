/* USER CODE BEGIN PFP */
void calculate_moments(float img[]){
    for(int c = 0; c < numCols; c++) {
        for (int r = 0; r < numRows; r++){
            for(int i = 0; i < 3; i ++){
                for(int j = 0; j < 3 - i; j++){
                    moments[i][j] += pow(c, i) * pow(r,j) * img[c * numRows + r];
                }
            }
        }
    }

    float centroid_x = moments[1][0] / moments[0][0];
    float centroid_y = moments[0][1] / moments[0][0];
    mu[1][1] = fmax(moments[1][1] - centroid_x * moments[0][1],0);
    mu[2][0] = fmax(moments[2][0] - centroid_x * moments[1][0],0);
    mu[0][2] = fmax(moments[0][2] - centroid_y * moments[0][1],0);
    mu[3][0] = fmax(moments[3][0] - 3 * centroid_x * moments[2][0] + 2 * pow(centroid_x, 2) * moments[1][0], 0);
    mu[2][1] = fmax(moments[2][1] - 2 * centroid_x * moments[1][1] - centroid_y * moments[2][0] + 2 * pow(centroid_x, 2) * moments[0][1],0);
    mu[1][2] = fmax(moments[1][2] - 2 * centroid_y * moments[1][1] - centroid_x * moments[0][2] + 2 * pow(centroid_y, 2) * moments[1][0],0);
    mu[0][3] = fmax(moments[0][3] - 3 * centroid_y * moments[0][2] + 2 * pow(centroid_y, 2) * moments[0][1], 0);
    nu[2][0] = mu[2][0] / pow(moments[0][0], 2);
    nu[1][1] = mu[1][1] / pow(moments[0][0],2);
    nu[0][2] = mu[0][2] / pow(moments[0][0], 2);
    nu[3][0] = mu[3][0] / pow(moments[0][0], 2.5);
    nu[2][1] = mu[2][1] / pow(moments[0][0], 2.5);
    nu[1][2] = mu[1][2] / pow(moments[0][0], 2.5);
    nu[0][3] = mu[0][3] / pow(moments[0][0], 2.5);
}

void calculate_hu_moments(){
hu_moments[0] = nu[2][0] + nu[0][2];
hu_moments[1] = pow(nu[2][0] - nu[0][2], 2) + 4 * pow(nu[1][1], 2);
hu_moments[2] = pow(nu[3][0] -3 * nu[1][2], 2) + pow(3 * nu[2][1] -nu[0][3], 2);
hu_moments[3] = pow(nu[3][0] + nu[1][2], 2) + pow(nu[2][1] + nu[0][3], 2);
hu_moments[4] = (nu[3][0] - 3 * nu[1][2])* (nu[3][0] + nu[1][2])* (pow(nu[3][0] + nu[1][2], 2) - 3 * pow(nu[2][1] + nu[0][3], 2)) + (3 * nu[2][1] - nu[0][3])* (nu[2][1] + nu[0][3])* (3 * pow(nu[3][0] + nu[1][2], 2) - pow(nu[2][1] + nu[0][3],2));
hu_moments[5] = (nu[2][0]-nu[0][2])* (pow(nu[3][0]+nu[1][2],2) - pow(nu[2][1] + nu[0][3],2)) + 4 * nu[1][1] * (nu[3][0] + nu[1][2]) * (nu[2][1]+nu[0][3]);
hu_moments[6] = (3 * nu[2][1] - nu[0][3])* (nu[3][0] + nu[1][2])* (pow(nu[3][0] + nu[1][2], 2) - 3 * pow(nu[2][1] + nu[0][3], 2))- (nu[3][0]-3 * nu[1][2]) * (nu[2][1]+nu[0][3]) * (3 * pow(nu[3][0]+nu[1][2],2)- pow(nu[2][1]+nu[0][3], 2));
}

/* USER CODE END PFP */