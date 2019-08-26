#include <stdio.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>

#define FF_INPUT_BUFFER_PADDING_SIZE 32
#define MV 1
#define RESIDUAL 2
static const char *filename = "origin_trans_b_1_b_g15.avi";

int decode_video(int gop_target, int pos_target, int representation, int accumulate){

//int decode_video(int gop_target, int pos_target, PyArrayObject ** bgr_arr, PyArrayObject ** mv_arr, PyArrayObject ** res_arr, int representation, int accumulate){
	printf("in decode_video\n");
	AVCodec *pCodec_MPEG4;
	//AVCodec *pCodec_H264;
	//AVCodec *pCodec_HEVC;

	AVCodecContext *pCodecCtx_MPEG4 = NULL;
	//AVCodecContext *pCodecCtx_H264 = NULL;
	//AVCodecContext *pCodecCtx_HEVC = NULL;
	AVCodecParserContext *pCodecParserCtx_MPEG4 = NULL;
	//AVCodecParserContext *pCodecParserCtx_H264 = NULL;
	//AVCodecParserContext *pCodecParserCtx_HEVC = NULL;
	FILE *fp_in;
	AVFrame *pFrame;
	AVFrame *pFrameBGR;
	AVFrame *bFrame;
	AVFrame *bFrameBGR;
	
	const int in_buffer_size = 4096;
	uint8_t in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE];
	memset(in_buffer + in_buffer_size, 0, FF_INPUT_BUFFER_PADDING_SIZE);

	uint8_t *cur_ptr;
	int cur_size;
	int cur_gop = -1;
	AVPacket packet;
	int ret, got_picture;

	avcodec_register_all();

	pCodec_MPEG4 = avcodec_find_decoder(AV_CODEC_ID_MPEG4);
	//pCodec_H264 = avcodec_find_decoder(AV_CODEC_ID_H264);
	//pCodec_HEVC = avcodec_find_decoder(AV_CODEC_ID_HEVC);
	
	if (!pCodec_MPEG4){// || !pCodec_H264 || !pCodec_HEVC) {
		printf("Codec not found\n");
		return -1;
	}
	pCodecCtx_MPEG4 = avcodec_alloc_context3(pCodec_MPEG4);
	//pCodecCtx_H264 = avcodec_alloc_context3(pCodec_H264);
	//pCodecCtx_HEVC = avcodec_alloc_context3(pCodec_HEVC);
	
	if (!pCodecCtx_MPEG4){// || !pCodecCtx_H264 || !pCodecCtx_HEVC) {
		printf("Could not allocate video codec context\n");
		return -1;
	}
	
	pCodecParserCtx_MPEG4 = av_parser_init(AV_CODEC_ID_MPEG4);
	//pCodecParserCtx_H264 = av_parser_init(AV_CODEC_ID_H264);
	//pCodecParserCtx_HEVC = av_parser_init(AV_CODEC_ID_HEVC);
	
	if (!pCodecParserCtx_MPEG4){// || !pCodecParserCtx_H264 || !pCodecParserCtx_HEVC) {
		printf("Could not allocate video parser context\n");
		return -1;
	}

	AVDictionary *opts = NULL;
	av_dict_set(&opts, "flags2", "+export_mvs", 0);
	if (avcodec_open2(pCodecCtx_MPEG4, pCodec_MPEG4, &opts) < 0){
		printf("Could not open codec\n");
		return -1;
	}

	fp_in = fopen(filename, "rb");
	if (!fp_in) {
		printf("Could not open input stream\n");
		return -1;
	}
	int cur_pos = 0;
	pFrame = av_frame_alloc();
	pFrameBGR = av_frame_alloc();
	bFrame = av_frame_alloc();
	bFrameBGR = av_frame_alloc();

	uint8_t *buffer;
	av_init_packet(&packet);
	int *accu_src = NULL;
	int *accu_src_old = NULL;
    int step = 0;
	while (1) {
		cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);
		if (cur_size == 0)
			break;
		cur_ptr = in_buffer;
        //printf("cursize: %d\n", cur_size);
		while (cur_size > 0) {
			int len = av_parser_parse2(pCodecParserCtx_MPEG4, pCodecCtx_MPEG4, &packet.data, &packet.size, cur_ptr, cur_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);
            //printf("len: %d", len);
			cur_ptr += len;
			cur_size -= len;

			if (packet.size == 0){//读进来了，但是现在数据不完整，出不来frame
                //printf("why im here\n");
                continue;
            }
			if (pCodecParserCtx_MPEG4 -> pict_type == AV_PICTURE_TYPE_I) {
				//printf("\n%d, I", step);
                printf("\nI");
                step ++;
				++cur_gop;
			}
			
			if (pCodecParserCtx_MPEG4 -> pict_type == AV_PICTURE_TYPE_P) {
				//printf("%d, P", step);
                printf("P");
                step ++;
			}
			if (pCodecParserCtx_MPEG4 -> pict_type == AV_PICTURE_TYPE_B) {
				//printf("%d, B", step);
                printf("B");
                step ++;
			}
            if (cur_gop == gop_target && cur_pos <= pos_target) {
      
                ret = avcodec_decode_video2(pCodecCtx_MPEG4, pFrame, &got_picture, &packet);  
                if (ret < 0) {  
                    printf("Decode Error.\n");  
                    return -1;  
                }
                int h = pFrame->height;
                int w = pFrame->width;
                //printf("lalalalalla\n %d %d\n", h, w);
            
            //printf("Got_Picture: %d\n", got_picture);
                if (got_picture) {
                    if ((cur_pos == 0              && accumulate  && representation == RESIDUAL) ||
                        (cur_pos == pos_target - 1 && !accumulate && representation == RESIDUAL) ||
                        cur_pos == pos_target) {
                        printf("\nCreate and load bgr\n");
                    }
                    if (representation == MV || representation == RESIDUAL) {
                        printf("\n Here\n");
                        AVFrameSideData *sd;
                        sd = av_frame_get_side_data(pFrame, AV_FRAME_DATA_MOTION_VECTORS);
                        if (sd) {
                            if (accumulate || cur_pos == pos_target) {
                                printf("\nCreate and load mv and residual\n");
                            }
                        }
                    }
                    cur_pos ++;
                }
            }
		}
	}

    packet.data = NULL;  
    packet.size = 0;  
    while(1){  
        ret = avcodec_decode_video2(pCodecCtx_MPEG4, pFrame, &got_picture, &packet);  
        printf("im here once, %d, %d\n", ret, got_picture);
        if (ret < 0) {  
            printf("Decode Error.\n");  
            return -1;  
        }  
        if (!got_picture){
            printf("im here\n");
            break;  
        }
    }  

    fclose(fp_in);

    av_parser_close(pCodecParserCtx_MPEG4);  
    //av_parser_close(pCodecParserCtx_H264);  
    //av_parser_close(pCodecParserCtx_HEVC);  
  
    av_frame_free(&pFrame);  
    av_frame_free(&pFrameBGR);  
    av_frame_free(&bFrame);  
    av_frame_free(&bFrameBGR);  
    avcodec_close(pCodecCtx_MPEG4);  
    //avcodec_close(pCodecCtx_H264);  
    //avcodec_close(pCodecCtx_HEVC);  
    av_free(pCodecCtx_MPEG4);
    //av_free(pCodecCtx_H264);
    //av_free(pCodecCtx_HEVC);

	return 0;
}
int main(int argc, char *argv[]){
/*	wchar_t *program = Py_DecodeLocate(argv[0], NULL);
	if (program == NULL){
		fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
		exit(1);
	}
	Py_SetProgramName(program);
	Py_Initialize();
	import_array();
*/
	/*PyArrayObject *bgr_arr = NULL;
	PyArrayObject *final_bgr_arr = NULL;
	PyArrayObject *mv_arr = NULL;
	PyArrayObject *res_arr = NULL;*/

	printf("hello, world!\n");
	if(decode_video(1, 2, 2, 1) < 0){
    //if(decode_video(0, 0, &bgr_arr, &mv_arr, &res_arr, 0, 0) < 0){
		printf("Decoding video failed.\n");
	}
//	PyMem_RawFree(program);
	return 0;
}
