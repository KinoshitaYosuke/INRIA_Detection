#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "svm.h"

#define YUDO_CD 0.9
#define YUDO_FD 0.995

using namespace std;

struct svm_node *x;
int max_nr_attr = 64;

double FD_max_ans = 0.0;
int FD_max_XY = 0;

struct svm_model* INRIA_CD;

struct svm_model* INRIA_24x72;

struct svm_model* INRIA_32x72;
struct svm_model* INRIA_32x80;
struct svm_model* INRIA_32x88;
struct svm_model* INRIA_32x96;
struct svm_model* INRIA_32x104;

struct svm_model* INRIA_40x72;
struct svm_model* INRIA_40x80;
struct svm_model* INRIA_40x88;
struct svm_model* INRIA_40x96;
struct svm_model* INRIA_40x104;
struct svm_model* INRIA_40x112;
struct svm_model* INRIA_40x120;
struct svm_model* INRIA_40x128;

struct svm_model* INRIA_48x88;
struct svm_model* INRIA_48x96;
struct svm_model* INRIA_48x104;
struct svm_model* INRIA_48x112;
struct svm_model* INRIA_48x120;
struct svm_model* INRIA_48x128;

struct svm_model* INRIA_56x104;
struct svm_model* INRIA_56x112;
struct svm_model* INRIA_56x120;
struct svm_model* INRIA_56x128;

struct svm_model* INRIA_64x120;
struct svm_model* INRIA_64x128;
struct svm_model* INRIA_64x120;

//static char *line = NULL;
static int max_line_len;

class Detect_Place {
public:
	int C_x;
	int C_y;
	int C_width;
	int C_height;
	float C_yudo;

	int F_x;
	int F_y;
	int F_width;
	int F_height;
	float F_yudo;

	int territory_num;

	int ratio_num;

	Detect_Place() {
		C_x = C_y = -1;
		C_width = C_height = -1;
		C_yudo = 0.0;

		F_x = F_y = -1;
		F_width = F_height = -1;
		F_yudo = 0.0;

		territory_num = -1;

		ratio_num = -1;
	}

};

void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}

double predict(float *hog_vector, int hog_dim, svm_model* Detector)
{
	int svm_type = svm_get_svm_type(Detector);
	int nr_class = svm_get_nr_class(Detector);
	double *prob_estimates = NULL;
	int j;

	int *labels = (int *)malloc(nr_class * sizeof(int));
	svm_get_labels(Detector, labels);
	prob_estimates = (double *)malloc(nr_class * sizeof(double));
	free(labels);


	max_line_len = 1024;
	//	line = (char *)malloc(max_line_len*sizeof(char));
	x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
	int i;
	double target_label, predict_label;
	char *idx, *val, *label, *endptr;
	int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0


							 //	for (i = 0; hog_vector[i] != NULL; i++)
	for (i = 0; i < hog_dim; i++)
	{
		//		clog << i << endl;
		if (i >= max_nr_attr - 1)	// need one more for index = -1
		{
			max_nr_attr *= 2;
			x = (struct svm_node *) realloc(x, max_nr_attr * sizeof(struct svm_node));
		}

		//			clog << i << endl;

		errno = 0;
		x[i].index = i;
		inst_max_index = x[i].index;

		errno = 0;
		x[i].value = hog_vector[i];
		//		cout << i << ":" << hog_vector[i] << endl;
	}
	x[i].index = -1;


	predict_label = svm_predict_probability(Detector, x, prob_estimates);
	//	if (prob_estimates[0] >= YUDO_CD)
	//		printf(" %f\n", prob_estimates[0]);

	free(x);
	//	free(line);

	return prob_estimates[0];
}

int minimum(int a, int b) {
	if (a<b) {
		return a;
	}
	return b;
}

void get_HOG(cv::Mat im, float* hog_vector) {
	int cell_size = 6;
	int rot_res = 9;
	int block_size = 3;
	int x, y, i, j, k, m, n, count;
	float dx, dy;
	float ***hist, *vec_tmp;
	float norm;
	CvMat *mag = NULL, *theta = NULL;
	//	FILE *hog_hist;

	//	fopen_s(&hog_hist,"d_im_hog.txt", "w");
	//	fopen_s(&hog_hist, "d_im_hog.bin", "w");
	//	fprintf(hog_hist, "%c ", '1');
	int counter = 1;

	mag = cvCreateMat(im.rows, im.cols, CV_32F);
	theta = cvCreateMat(im.rows, im.cols, CV_32F);
	for (y = 0; y<im.rows; y++) {
		for (x = 0; x<im.cols; x++) {
			if (x == 0 || x == im.cols - 1 || y == 0 || y == im.rows - 1) {
				cvmSet(mag, y, x, 0.0);
				cvmSet(theta, y, x, 0.0);
			}
			else {
				dx = double((uchar)im.data[y*im.step + x + 1]) - double((uchar)im.data[y*im.step + x - 1]);
				dy = double((uchar)im.data[(y + 1)*im.step + x]) - double((uchar)im.data[(y - 1)*im.step + x]);
				cvmSet(mag, y, x, sqrt(dx*dx + dy * dy));
				cvmSet(theta, y, x, atan(dy / (dx + 0.01)));
			}
		}
	}

	// histogram generation for each cell
	hist = (float***)malloc(sizeof(float**) * (int)ceil((float)im.rows / (float)cell_size));
	if (hist == NULL) exit(1);
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		hist[i] = (float**)malloc(sizeof(float*)*(int)ceil((float)im.cols / (float)cell_size));
		if (hist[i] == NULL) exit(1);
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++) {
			hist[i][j] = (float *)malloc(sizeof(float)*rot_res);
			if (hist[i][j] == NULL) exit(1);
		}
	}
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++) {
			for (k = 0; k<rot_res; k++) {
				hist[i][j][k] = 0.0;
			}
		}
	}
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size); j++) {
			for (m = i * cell_size; m<minimum((i + 1)*cell_size, im.rows); m++) {
				for (n = j * cell_size; n<minimum((j + 1)*cell_size, im.cols); n++) {
					hist[i][j][(int)floor((cvmGet(theta, m, n) + CV_PI / 2)*rot_res / CV_PI)] += cvmGet(mag, m, n);
				}
			}
		}
	}

	// normalization for each block & generate vector
	vec_tmp = (float *)malloc(sizeof(float)*block_size*block_size*rot_res);
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size) - (block_size - 1); i++) {
		for (j = 0; j<(int)ceil((float)im.cols / (float)cell_size) - (block_size - 1); j++) {
			count = 0;
			norm = 0.0;
			for (m = i; m<i + block_size; m++) {
				for (n = j; n<j + block_size; n++) {
					for (k = 0; k<rot_res; k++) {
						vec_tmp[count++] = hist[m][n][k];
						norm += hist[m][n][k] * hist[m][n][k];
					}
				}
			}
			for (count = 0; count<block_size*block_size*rot_res; count++) {
				vec_tmp[count] = vec_tmp[count] / (sqrt(norm + 1));
				if (vec_tmp[count]>0.2) vec_tmp[count] = 0.2;
				//		fprintf(hog_hist, "%d:%.4f ", counter, vec_tmp[count]);
				hog_vector[counter] = vec_tmp[count];
				//		cout << counter << ":" << hog_vector[counter] << endl;
				//		printf("%d:%.4f ",counter, vec_tmp[count]);
				counter++;
			}
		}
	}
	//	printf("\n");
	//	fprintf(hog_hist, "\n");
	for (i = 0; i<(int)ceil((float)im.rows / (float)cell_size); i++) {
		for (j = 0; j <(int)ceil((float)im.cols / (float)cell_size); j++) {
			free(hist[i][j]);
		}
		free(hist[i]);
	}
	free(hist);
	cvReleaseMat(&mag);
	cvReleaseMat(&theta);
	//	fclose(hog_hist);
}

cv::Mat draw_rectangle(cv::Mat ans_im, int x, int y, int width, int height, int r, int g, int b) {
	rectangle(ans_im, cvPoint(x, y), cvPoint(x + width, y + height), CV_RGB(r, g, b), 1);
	return ans_im;
}

int dimension(int x, int y) {

	return (int)(81 * ((int)ceil((float)x / 6) - 2) * ((int)ceil((float)y / 6) - 2));

	if (x % 6 == 0) {
		return 81 * (x / 6 - 2) * (y / 6 - 1);
	}
	if (y % 6 == 0) {
		return 81 * (x / 6 - 1) * (y / 6 - 2);
	}
	if (x % 6 == 0 && y % 6 == 0) {
		return 81 * (x / 6 - 2) * (y / 6 - 2);
	}
	else
		return 81 * (x / 6 - 1) * (y / 6 - 1);
}

void FD_predict(int width, int height, cv::Mat FD_img, svm_model* Detector) {
	cv::Mat FD_cut_im(FD_img, cv::Rect(32 - width / 2, 64 - height / 2, width, height));
	//	cv::Mat FD_max_img = FD_cut_im;
	//	cvtColor(FD_cut_im, FD_cut_im, CV_RGB2GRAY);
	int hog_dim = dimension(FD_cut_im.cols, FD_cut_im.rows);
	float FD_vector[20000];	//各次元のHOGを格納
	get_HOG(FD_cut_im, FD_vector);	//HOGの取得

	double ans = predict(FD_vector, hog_dim, Detector);

	//	cout << "x,y:" << width << "," << height << "=" << ans << endl;

	if (FD_max_ans < ans && YUDO_FD < ans) {
		FD_max_ans = ans;
		FD_max_XY = width * 1000 + height;
		//	cv::imwrite("result_FD.bmp", FD_cut_im);
	}
}

int main(int argc, char** argv) {
	//変数宣言
	//	int x, y;

	int file_num = 1;

	int count = 0;
	int hog_dim;

	//CoarseDetectorの取り込み
	if ((INRIA_CD = svm_load_model("C:/model_file/pre_model/INRIA_CD.model")) == 0)exit(1);

	if ((INRIA_24x72 = svm_load_model("C:/model_file/pre_model/INRIA_24x72.model")) == 0)exit(1);

	if ((INRIA_32x72 = svm_load_model("C:/model_file/pre_model/INRIA_32x72.model")) == 0)exit(1);
	if ((INRIA_32x80 = svm_load_model("C:/model_file/pre_model/INRIA_32x80.model")) == 0)exit(1);
	if ((INRIA_32x88 = svm_load_model("C:/model_file/pre_model/INRIA_32x88.model")) == 0)exit(1);
	if ((INRIA_32x96 = svm_load_model("C:/model_file/pre_model/INRIA_32x96.model")) == 0)exit(1);
	if ((INRIA_32x104 = svm_load_model("C:/model_file/pre_model/INRIA_32x104.model")) == 0)exit(1);

	if ((INRIA_40x72 = svm_load_model("C:/model_file/pre_model/INRIA_40x72.model")) == 0)exit(1);
	if ((INRIA_40x80 = svm_load_model("C:/model_file/pre_model/INRIA_40x80.model")) == 0)exit(1);
	if ((INRIA_40x88 = svm_load_model("C:/model_file/pre_model/INRIA_40x88.model")) == 0)exit(1);
	if ((INRIA_40x96 = svm_load_model("C:/model_file/pre_model/INRIA_40x96.model")) == 0)exit(1);
	if ((INRIA_40x104 = svm_load_model("C:/model_file/pre_model/INRIA_40x104.model")) == 0)exit(1);
	if ((INRIA_40x112 = svm_load_model("C:/model_file/pre_model/INRIA_40x112.model")) == 0)exit(1);
	if ((INRIA_40x120 = svm_load_model("C:/model_file/pre_model/INRIA_40x120.model")) == 0)exit(1);
	if ((INRIA_40x128 = svm_load_model("C:/model_file/pre_model/INRIA_40x128.model")) == 0)exit(1);

	if ((INRIA_48x88 = svm_load_model("C:/model_file/pre_model/INRIA_48x88.model")) == 0)exit(1);
	if ((INRIA_48x96 = svm_load_model("C:/model_file/pre_model/INRIA_48x96.model")) == 0)exit(1);
	if ((INRIA_48x104 = svm_load_model("C:/model_file/pre_model/INRIA_48x104.model")) == 0)exit(1);
	if ((INRIA_48x112 = svm_load_model("C:/model_file/pre_model/INRIA_48x112.model")) == 0)exit(1);
	if ((INRIA_48x120 = svm_load_model("C:/model_file/pre_model/INRIA_48x120.model")) == 0)exit(1);
	if ((INRIA_48x128 = svm_load_model("C:/model_file/pre_model/INRIA_48x128.model")) == 0)exit(1);

	if ((INRIA_56x104 = svm_load_model("C:/model_file/pre_model/INRIA_56x104.model")) == 0)exit(1);
	if ((INRIA_56x112 = svm_load_model("C:/model_file/pre_model/INRIA_56x112.model")) == 0)exit(1);
	if ((INRIA_56x120 = svm_load_model("C:/model_file/pre_model/INRIA_56x120.model")) == 0)exit(1);
	if ((INRIA_56x128 = svm_load_model("C:/model_file/pre_model/INRIA_56x128.model")) == 0)exit(1);

	if ((INRIA_64x120 = svm_load_model("C:/model_file/pre_model/INRIA_64x120.model")) == 0)exit(1);
	if ((INRIA_64x128 = svm_load_model("C:/model_file/pre_model/INRIA_64x128.model")) == 0)exit(1);

	//テスト画像ファイル一覧メモ帳読み込み
	char test_name[1024];
	FILE *test_data, *result_data;
	if (fopen_s(&test_data, "test_list.txt", "r") != 0) {
		cout << "missing" << endl;
		return 0;
	}


	while (fgets(test_name, 256, test_data) != NULL) {
		if (fopen_s(&result_data, "result_data.txt", "a") != 0) {
			cout << "missing 2" << endl;
			return 0;
		}

		string name_tes = test_name;
		char new_test_name[1024];
		for (int i = 0; i < name_tes.length() - 1; i++) {
			new_test_name[i] = test_name[i];
			new_test_name[i + 1] = '\0';
		}
		count = 0;

		char test_path[1024] = "C:/photo/test_data_from_demo/test_data/";
		char result_path[1024] = "result_data/";
		char binary_path[1024] = "result_binary/";

		strcat_s(test_path, new_test_name);
		strcat_s(result_path, new_test_name);

		fprintf_s(result_data, new_test_name);

		for (int i = 0; i < 1024; i++) {
			if (new_test_name[i] == 'b') {
				new_test_name[i] = 'j';
				new_test_name[i + 1] = 'p';
				new_test_name[i + 2] = 'g';
				new_test_name[i + 3] = '\0';
				break;
			}
			else new_test_name[i] = new_test_name[i];
		}
		strcat_s(binary_path, new_test_name);
		cout << binary_path << endl;


		//画像の取り込み
		cv::Mat ans_img_CF = cv::imread(new_test_name, 1);	//検出する画像
//		cv::Mat ans_img_CF = cv::imread("Sun_Nov_26_14_02_00_95.bmp", 1);	//検出する画像
		cv::Mat res_bin = cv::Mat::zeros(ans_img_CF.rows, ans_img_CF.cols, CV_8UC3);
		cv::Mat check_img = ans_img_CF.clone();
		
		//リザルトファイルに画像ファイル名を書き込み
		fprintf_s(result_data, new_test_name);

		cout << file_num << ":" << new_test_name << endl;
		file_num++;

		//Detect_Placeオブジェクトの作成
		Detect_Place detect[300];

		//Coarse Detectorによる人物検出
		cv::Mat CD_img[300];
		
		float normalize_num[15] = { 132,192, 256, 320, 384, -1 };

		for (int img_size = 0; normalize_num[img_size] != -1; img_size++) {
			cv::Mat img, dst;			//検出矩形処理を施す画像
			cvtColor(ans_img_CF, img, CV_RGB2GRAY);
			cvtColor(ans_img_CF, dst, CV_RGB2GRAY);
			cv::resize(img, img, cv::Size(), normalize_num[img_size] / img.rows, normalize_num[img_size] / img.rows, CV_INTER_LINEAR);
			for (int y = 2; (y +132) <= img.rows; y += 8) {
				for (int x = 2; (x + 68) <= img.cols; x += 8) {
					cv::Mat d_im(img, cv::Rect(x, y, 64, 128));
					hog_dim = dimension(d_im.cols, d_im.rows);
					float hog_vector[20000];							//各次元のHOGを格納
					get_HOG(d_im, hog_vector);	//HOGの取得
					double ans = predict(hog_vector, hog_dim, INRIA_CD);	//尤度の算出
					if (ans >= YUDO_CD) {//尤度から人物か非人物かの判断
						detect[count].C_yudo = ans;
						detect[count].C_x = x * ans_img_CF.rows / normalize_num[img_size];
						detect[count].C_y = y * ans_img_CF.rows / normalize_num[img_size];
						detect[count].C_width = 64 * ans_img_CF.rows / normalize_num[img_size];
						detect[count].C_height = 128 * ans_img_CF.rows / normalize_num[img_size];
						detect[count].ratio_num = img_size;
						CD_img[count] = img(cv::Rect(x - 2, y - 2, 68, 132));
				//		CD_img[count] = dst(cv::Rect(x - 2, y - 2, 132, 132));
						check_img = draw_rectangle(check_img, detect[count].C_x, detect[count].C_y, detect[count].C_width, detect[count].C_height, 255, 0, 0);
						count++;
					}
				}
			}
		}
		
		//Fine Detectorによる検出
		for (int i = 0; i < count; i++) {
			float zure_yudo[25];
			int zure_count = 0;
			FD_max_ans = 0.0;
			FD_max_XY = 0;

			for (int a = 0; a <= 4; a++) {
				for (int b = 0; b <= 4; b++) {

					cv::Mat FD_img = CD_img[i](cv::Rect(a, b, 64, 128));
					FD_predict(24, 72, FD_img, INRIA_24x72);

					FD_predict(32, 80, FD_img, INRIA_32x80);
					FD_predict(32, 88, FD_img, INRIA_32x88);
					FD_predict(32, 96, FD_img, INRIA_32x96);
					FD_predict(32, 104, FD_img, INRIA_32x104);
					
					FD_predict(40, 72, FD_img, INRIA_40x72);
					FD_predict(40, 80, FD_img, INRIA_40x80);
					FD_predict(40, 88, FD_img, INRIA_40x88);
					FD_predict(40, 96, FD_img, INRIA_40x96);
					FD_predict(40, 104, FD_img, INRIA_40x104);
					FD_predict(40, 112, FD_img, INRIA_40x112);
					FD_predict(40, 120, FD_img, INRIA_40x120);
					FD_predict(40, 128, FD_img, INRIA_40x128);

					FD_predict(48, 88, FD_img, INRIA_48x88);
					FD_predict(48, 96, FD_img, INRIA_48x96);
					FD_predict(48, 104, FD_img, INRIA_48x104);
					FD_predict(48, 112, FD_img, INRIA_48x112);
					FD_predict(48, 120, FD_img, INRIA_48x120);
					FD_predict(48, 128, FD_img, INRIA_48x128);

					FD_predict(56, 104, FD_img, INRIA_56x104);
					FD_predict(56, 112, FD_img, INRIA_56x112);
					FD_predict(56, 120, FD_img, INRIA_56x120);
					FD_predict(56, 128, FD_img, INRIA_56x128);

					FD_predict(64, 120, FD_img, INRIA_64x120);
					FD_predict(64, 128, FD_img, INRIA_64x128);

					if (FD_max_ans > YUDO_FD && FD_max_ans > detect[i].F_yudo) {
						detect[i].F_yudo = FD_max_ans;
						detect[i].F_width = FD_max_XY / 1000;
						detect[i].F_height = FD_max_XY % 1000;
						zure_count = a * 10 + b;
					}
				}
			}

			if (zure_count != 0) {
				detect[i].C_x += (zure_count / 10) - 2;
				detect[i].C_y += (zure_count % 10) - 2;

				int F_x1 = (detect[i].C_x + 32 - detect[i].F_width / 2) * ans_img_CF.rows / normalize_num[detect[i].ratio_num];
				int F_y1 = (detect[i].C_y + 64 - detect[i].F_height / 2) * ans_img_CF.rows / normalize_num[detect[i].ratio_num];
				int F_width = detect[i].F_width * ans_img_CF.rows / normalize_num[detect[i].ratio_num];
				int F_height = detect[i].F_height * ans_img_CF.rows / normalize_num[detect[i].ratio_num];

				detect[i].F_x = F_x1;
				detect[i].F_y = F_y1;
				detect[i].F_width = F_width;
				detect[i].F_height = F_height;

				check_img = draw_rectangle(check_img, detect[i].C_x, detect[i].C_y, detect[i].C_width, detect[i].C_height, 255, 0, 0);
				check_img = draw_rectangle(check_img, detect[i].F_x, detect[i].F_y, detect[i].F_width, detect[i].F_height, 0, 255, 0);
				cout << "FD:(" << F_x1 << "," << F_y1 << "),(" << F_width << "," << F_height << "):" << F_width*F_height << ", " << detect[i].F_yudo << endl;;
			}
		}

		//領域の統一
		int t_num = 0;
		for (int n = 0; detect[n].C_yudo != 0; n++) {
			if (detect[n].F_yudo == 0) continue;
			if (detect[n].territory_num == -1) {
				t_num++;
				detect[n].territory_num = t_num;
			}
			for (int m = n + 1; detect[m].C_yudo != 0; m++) {
				if (detect[m].F_yudo == 0) continue;

				if ((detect[n].C_x + detect[n].C_width/2) - 100 <= (detect[m].C_x + detect[m].C_width/2)
					&& (detect[m].C_x + detect[m].C_width/2) <= (detect[n].C_x + detect[n].C_width/2) + 100
					&&
					(detect[n].C_y + detect[n].C_height/2) - 100 <= (detect[m].C_y + detect[m].C_height/2)
					&& (detect[m].C_y + detect[m].C_height/2) <= (detect[n].C_y + detect[n].C_height/2) + 100) {

					detect[m].territory_num = detect[n].territory_num;
				}
			}

		}
		//統一領域ごとに検出結果の表示
		for (int i = 1; i <= t_num; i++) {
			int final_num = 0;
			float fyudo = 0, cyudo = 0;
			int area = 0;
			for (int k = 0; detect[k].C_yudo != 0; k++) {
		//		int F_x1 = (detect[k].C_x + 32 - detect[k].F_width / 2) * 480 / normalize_num[detect[k].ratio_num];
		//		int F_y1 = (detect[k].C_y + 32 - detect[k].F_height / 2) * 480 / normalize_num[detect[k].ratio_num];
		//		int F_width = detect[k].F_width * 480 / normalize_num[detect[k].ratio_num];
		//		int F_height = detect[k].F_height * 480 / normalize_num[detect[k].ratio_num];
	/*			if (detect[k].territory_num == i && (detect[k].F_yudo+detect[k].C_yudo) > (fyudo+cyudo)) {
					final_num = k;
					fyudo = detect[k].F_yudo;
					cyudo = detect[k].C_yudo;

				}
				*/
				if (detect[k].territory_num == i) {
					if ((detect[k].F_width * detect[k].F_height) > area) {
						final_num = k;
						fyudo = detect[k].F_yudo;
						area = detect[k].F_width * detect[k].F_height;
					}
					else if ((detect[k].F_width*detect[k].F_height) == area && detect[k].F_yudo > fyudo) {
						final_num = k;
						fyudo = detect[k].F_yudo;
					}
				}
			}

			cout << "area" << area << endl;

			//矩形表示
			//CDの矩形
			ans_img_CF = draw_rectangle(ans_img_CF, detect[final_num].C_x, detect[final_num].C_y, detect[final_num].C_width, detect[final_num].C_height, 255, 0, 0);
			//FDの矩形
			ans_img_CF = draw_rectangle(ans_img_CF, detect[final_num].F_x, detect[final_num].F_y, detect[final_num].F_width, detect[final_num].F_height, 0, 255, 0);
			//FD結果をテキストファイルに保存
			fprintf_s(result_data, ", %d, %d, %d, %d", detect[final_num].F_x, detect[final_num].F_y, detect[final_num].F_width, detect[final_num].F_height);

			for (int n = detect[final_num].F_y; n < detect[final_num].F_y + detect[final_num].F_height; n++) {
				for (int m = detect[final_num].F_x; m < detect[final_num].F_x + detect[final_num].F_width; m++) {
					res_bin.at<cv::Vec3b>(n, m) = cv::Vec3b(255, 255, 255);
				}
			}

		}

		//テキストファイル改行
		fprintf_s(result_data, "\n");

		cv::cvtColor(res_bin, res_bin, CV_RGB2GRAY);
		cv::imwrite(binary_path, res_bin);
		
		//画像の保存(検出ができていてもいなくても保存)
		cv::imwrite(result_path, ans_img_CF);

		fclose(result_data);
	}
	fclose(test_data);

	return 0;
}
