#include "com_binhnn_project2_facerecognition_Process.h"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/clustering.h>
#include <iostream>
#include <string.h>


using namespace std;
using namespace dlib;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
        alevel0<
                alevel1<
                        alevel2<
                                alevel3<
                                        alevel4<
                                                max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
                                                        input_rgb_image_sized<150>
                                                >>>>>>>>>>>>;


int len =0;

int * faceRecognition(cv::Mat &temp)
{
    int *result = nullptr;

    try{

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("/storage/emulated/0/shape_predictor_5_face_landmarks.dat") >> sp;
        anet_type net;
        deserialize("/storage/emulated/0/dlib_face_recognition_resnet_model_v1.dat") >> net;

        typedef matrix<float, 0, 1> sample_type;
        typedef radial_basis_kernel<sample_type> kernel_type;

        typedef decision_function<kernel_type> dec_funct_type;
        typedef normalized_function<dec_funct_type> funct_type;

        funct_type learned_function;
        deserialize("/storage/emulated/0/saved_function.dat")  >> learned_function;

        std::vector<matrix<rgb_pixel>> faces;
        cv_image<bgr_pixel> img(temp);

        std::vector<int> vStartX, vStartY, vEndX, vEndY;
        std::vector<dlib::rectangle> facerects = detector(img);
        for (auto face : facerects)
        {
            auto shape = sp(img, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
            faces.push_back(move(face_chip));
            vStartX.push_back(face.left());
            vStartY.push_back(face.top());
            vEndX.push_back(face.right());
            vEndY.push_back(face.bottom());
        }
        len = facerects.size() * 5;
        if(len != 0)
            result = new int(len);
        else
            return nullptr;
        std::vector<sample_type> face_descriptors = net(faces);
        for (int i = 0; i < face_descriptors.size(); ++i)
        {
            int label;

            if (learned_function(face_descriptors[i]) > 0)
            {
                label = 1;
            }
            else
                label = 0;

            *(result+i*5)= vStartX[i];
            *(result+i*5+1)= vStartY[i];
            *(result+i*5+2)= vEndX[i];
            *(result+i*5+3)= vEndY[i];
            *(result+i*5+4)= label;
            //cv::putText(temp, "Other", (LeftBotForText[i]), cv::FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv::LINE_AA);
        }
    }catch(serialization_error& e){
        std::cout<<endl<<e.what()<<endl;
    }
    return result;
}


extern "C"
JNIEXPORT jintArray JNICALL Java_com_binhnn_project2_1facerecognition_Process_detectFace(JNIEnv *env, jclass type,
                                                             jlong addrInput) {
    cv::Mat& inputMat = *(cv::Mat*)addrInput;
    int *result = faceRecognition(inputMat);
    if(result == nullptr)
        return nullptr;
    else {
        jintArray jresult = env->NewIntArray(len );
        env->SetIntArrayRegion(jresult, 0, len, &result[0]);
        return jresult;
    }
}