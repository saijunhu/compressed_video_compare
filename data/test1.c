#include <Python.h>
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdio.h>
#include <omp.h>

#include <libavutil/motion_vector.h>
#include <libavformat/avformat.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>

#define FF_INPUT_BUFFER_PADDING_SIZE 32

static PyObject *CoviexinfoError;//used by PyInit_coviexinfo()

static AVFormatContext *fmt_ctx = NULL;
static AVCodecContext *video_dec_ctx = NULL;
static AVStream *video_stream = NULL;
static const char *src_filename = NULL;

static int video_stream_idx = -1;
static AVFrame *frame = NULL;
static int video_frame_count = 0;


static int decode_packet(const AVPacket *pkt)
{
    int ret = avcodec_send_packet(video_dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error while sending a packet to the decoder: %s\n", av_err2str(ret));
        return ret;
    }

    while (ret >= 0)  {
        ret = avcodec_receive_frame(video_dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            fprintf(stderr, "Error while receiving a frame from the decoder: %s\n", av_err2str(ret));
            return ret;
        }

        if (ret >= 0) {
            //int i;
            AVFrameSideData *sd;

            video_frame_count++;
            sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
            /*
            if (sd) {
                const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
                for (i = 0; i < sd->size / sizeof(*mvs); i++) {
                    const AVMotionVector *mv = &mvs[i];
                    printf("%d,%2d,%2d,%2d,%4d,%4d,%4d,%4d,0x%"PRIx64"\n",
                        video_frame_count, mv->source,
                        mv->w, mv->h, mv->src_x, mv->src_y,
                        mv->dst_x, mv->dst_y, mv->flags);
                }

            }
            */
            av_frame_unref(frame);
        }
    }

    return 0;
}

static int open_codec_context(AVFormatContext *fmt_ctx, enum AVMediaType type)
{
    int ret;
    AVStream *st;
    AVCodecContext *dec_ctx = NULL;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, &dec, 0);
    if (ret < 0) {
        fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(type), src_filename);
        return ret;
    } else {
        int stream_idx = ret;
        st = fmt_ctx->streams[stream_idx];

        dec_ctx = avcodec_alloc_context3(dec);
        if (!dec_ctx) {
            fprintf(stderr, "Failed to allocate codec\n");
            return AVERROR(EINVAL);
        }

        ret = avcodec_parameters_to_context(dec_ctx, st->codecpar);
        if (ret < 0) {
            fprintf(stderr, "Failed to copy codec parameters to codec context\n");
            return ret;
        }

        /* Init the video decoder */
        av_dict_set(&opts, "flags2", "+export_mvs", 0);
        if ((ret = avcodec_open2(dec_ctx, dec, &opts)) < 0) {
            fprintf(stderr, "Failed to open %s codec\n",
                    av_get_media_type_string(type));
            return ret;
        }

        video_stream_idx = stream_idx;
        video_stream = fmt_ctx->streams[video_stream_idx];
        video_dec_ctx = dec_ctx;
    }

    return 0;
}



static PyObject *get_num_frames(PyObject *self, PyObject *args)
{
	int ret = 0;
    AVPacket pkt = { 0 };
    if (!PyArg_ParseTuple(args, "s", &src_filename)) return NULL;
    if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0) {
        fprintf(stderr, "Could not open source file %s\n", src_filename);
        exit(1);
    }
    //printf("fmt_ctx->filename = %s\n",fmt_ctx->filename );
    
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
        fprintf(stderr, "Could not find stream information\n");
        exit(1);
    }
    //printf("%s\n",dir_name );
    //video_dec_ctx->dir_path=dir_name;

    open_codec_context(fmt_ctx, AVMEDIA_TYPE_VIDEO);
    //video_dec_ctx在open_codec_context()函数中已经初始化，故可以对其赋值。
    //video_dec_ctx->dir_path=dir_name;

    av_dump_format(fmt_ctx, 0, src_filename, 0);

    if (!video_stream) {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        ret = 1;
        goto end;
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate frame\n");
        ret = AVERROR(ENOMEM);
        goto end;
    }

    int frame_count=0;
    /* read frames from the file */
    while (av_read_frame(fmt_ctx, &pkt) >= 0) {
        if (pkt.stream_index == video_stream_idx){
            ret = decode_packet(&pkt);
            frame_count++;
            //printf("Index_frame=%d\n",Index_frame );
            //printf("frame_order= %d\n", video_dec_ctx->frame_order);
        }
        av_packet_unref(&pkt);
        if (ret < 0)
            break;
    }

    /* flush cached frames */
    decode_packet(NULL);

end:
    avcodec_free_context(&video_dec_ctx);
    avformat_close_input(&fmt_ctx);
    av_frame_free(&frame);
    //return ret < 0;
    return Py_BuildValue("i", frame_count);
}

static PyObject *extract(PyObject *self, PyObject *args)
{
	PyArrayObject *mv_ref_arr = NULL;
	PyArrayObject *depth_qp_arr=NULL;
	PyArrayObject *residual_Y=NULL;
	PyArrayObject *residual_UV=NULL;
	int ret = 0;
	int Index_frame=0,representation,num_frames;
    AVPacket pkt = { 0 };
    if (!PyArg_ParseTuple(args, "sii", &src_filename,&representation,&num_frames)) return NULL;

    if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0) {
        fprintf(stderr, "Could not open source file %s\n", src_filename);
        exit(1);
    }
    //printf("fmt_ctx->filename = %s\n",fmt_ctx->filename );
    
    if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
        fprintf(stderr, "Could not find stream information\n");
        exit(1);
    }
    //printf("%s\n",dir_name );
    //video_dec_ctx->dir_path=dir_name;

    open_codec_context(fmt_ctx, AVMEDIA_TYPE_VIDEO);
    //video_dec_ctx在open_codec_context()函数中已经初始化，故可以对其赋值。
    //video_dec_ctx->dir_path=dir_name;

    av_dump_format(fmt_ctx, 0, src_filename, 0);

    if (!video_stream) {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        ret = 1;
        goto end;
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate frame\n");
        ret = AVERROR(ENOMEM);
        goto end;
    }

    //printf("framenum,source,blockw,blockh,srcx,srcy,dstx,dsty,flags\n");
    int mv_ref_h=video_dec_ctx->height/4;
    int mv_ref_w=video_dec_ctx->width/4;
    npy_intp dims[3];
    dims[0] = mv_ref_h;
    dims[1] = mv_ref_w;
    dims[2] = 6*num_frames;
    mv_ref_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0);

    int depth_qp_h=video_dec_ctx->height/16;
    int depth_qp_w=video_dec_ctx->width/16;
    npy_intp depth_qp_dims[3];
    depth_qp_dims[0] = depth_qp_h;
    depth_qp_dims[1] = depth_qp_w;
    depth_qp_dims[2] = 2*num_frames;
    depth_qp_arr = PyArray_ZEROS(3, depth_qp_dims, NPY_INT32, 0);

    int residual_Y_h=video_dec_ctx->height;
    int residual_Y_w=video_dec_ctx->width;
    npy_intp residual_Y_dims[3];
    residual_Y_dims[0] = residual_Y_h;
    residual_Y_dims[1] = residual_Y_w;
    residual_Y_dims[2] = num_frames;
    residual_Y = PyArray_ZEROS(3, residual_Y_dims, NPY_INT32, 0);

    int residual_UV_h=video_dec_ctx->height/2;
    int residual_UV_w=video_dec_ctx->width/2;
    npy_intp residual_UV_dims[3];
    residual_UV_dims[0] = residual_UV_h;
    residual_UV_dims[1] = residual_UV_w;
    residual_UV_dims[2] = 2*num_frames;
    residual_UV = PyArray_ZEROS(3, residual_UV_dims, NPY_INT32, 0);
    
    /* read frames from the file */
    while (av_read_frame(fmt_ctx, &pkt) >= 0) {
        if (pkt.stream_index == video_stream_idx){
            ret = decode_packet(&pkt);
            if(representation==0){//representation==0 denotes extracting mvs and ref.
            	for(int i=0;i<mv_ref_h;i++){
	            	for(int j=0;j<mv_ref_w;j++){

	            		*((int32_t*)PyArray_GETPTR3(mv_ref_arr, i, j,Index_frame*6)) = video_dec_ctx->mv_0_x_cache[i][j];
	            		*((int32_t*)PyArray_GETPTR3(mv_ref_arr, i, j,Index_frame*6+1)) = video_dec_ctx->mv_0_y_cache[i][j];
	            		*((int32_t*)PyArray_GETPTR3(mv_ref_arr, i, j,Index_frame*6+2)) = video_dec_ctx->ref_0_cache[i][j];
	            		*((int32_t*)PyArray_GETPTR3(mv_ref_arr, i, j,Index_frame*6+3)) = video_dec_ctx->mv_1_x_cache[i][j];
	            		*((int32_t*)PyArray_GETPTR3(mv_ref_arr, i, j,Index_frame*6+4)) = video_dec_ctx->mv_1_y_cache[i][j];
	            		*((int32_t*)PyArray_GETPTR3(mv_ref_arr, i, j,Index_frame*6+5)) = video_dec_ctx->ref_1_cache[i][j];
	            		//printf("%d ",video_dec_ctx->mv_0_x_cache[i][j] );
	            	}
	            	//printf("\n");
            	}
            //printf("\n\n");
            }else if(representation==1){//representation==1 denotes extracting depth and qp.
            	for(int i=0;i<depth_qp_h;i++){
	            	for(int j=0;j<depth_qp_w;j++){

	            		*((int32_t*)PyArray_GETPTR3(depth_qp_arr, i, j,Index_frame*2)) = video_dec_ctx->depth[i][j];
	            		*((int32_t*)PyArray_GETPTR3(depth_qp_arr, i, j,Index_frame*2+1)) = video_dec_ctx->QP_Table[i][j];
	            	}
	            	//printf("\n");
            	}
            }else if(representation==2){//representation==2 denotes extracting residual Y.
            	for(int i=0;i<residual_Y_h;i++){
	            	for(int j=0;j<residual_Y_w;j++){
	            		*((int32_t*)PyArray_GETPTR3(residual_Y, i, j,Index_frame)) = video_dec_ctx->Last_Y[i][j]-video_dec_ctx->Pred_Y[i][j];
	            	}
	            	//printf("\n");
            	}
            }else if(representation==3){//representation==3 denotes extracting residual UV.
            	for(int i=0;i<residual_UV_h;i++){
	            	for(int j=0;j<residual_UV_w;j++){
	            		*((int32_t*)PyArray_GETPTR3(residual_UV, i, j,Index_frame*2)) = video_dec_ctx->Last_U[i][j]-video_dec_ctx->Pred_U[i][j];
	            		*((int32_t*)PyArray_GETPTR3(residual_UV, i, j,Index_frame*2+1)) = video_dec_ctx->Last_V[i][j]-video_dec_ctx->Pred_V[i][j];
	            	}
	            	//printf("\n");
            	}
            }
            
            Index_frame++;
            //printf("Index_frame=%d\n",Index_frame );
            //printf("frame_order= %d\n", video_dec_ctx->frame_order);


        }
        av_packet_unref(&pkt);
        if (ret < 0)
            break;
    }

    /* flush cached frames */
    decode_packet(NULL);

    //给mv_arr赋值
    /*
    int h=3,w=4;
    npy_intp dims[3];
    dims[0] = h;
    dims[1] = w;
    dims[2] = 5;
    mv_arr = PyArray_ZEROS(3, dims, NPY_INT32, 0);
    for(int n=0;n<5;n++){
    	for(int i=0;i<h;i++){
	    	for(int j=0;j<w;j++){
	    		*((int32_t*)PyArray_GETPTR3(mv_arr, i, j,n)) = 5;
	    	}
	    }
    }
    */
    

    //int frame_count=110;
    //return Py_BuildValue("i", Index_frame);
end:
    avcodec_free_context(&video_dec_ctx);
    avformat_close_input(&fmt_ctx);
    av_frame_free(&frame);
    //return Py_BuildValue("i", Index_frame);
    if(representation==0){
    	Py_XDECREF(depth_qp_arr);
    	Py_XDECREF(residual_Y);
    	Py_XDECREF(residual_UV);
    	return mv_ref_arr;
    }else if(representation==1){
    	Py_XDECREF(mv_ref_arr);
    	Py_XDECREF(residual_Y);
    	Py_XDECREF(residual_UV);
    	return depth_qp_arr;
    }else if(representation==2){
    	Py_XDECREF(mv_ref_arr);
    	Py_XDECREF(depth_qp_arr);
    	Py_XDECREF(residual_UV);
    	return residual_Y;
    }else if(representation==3){
    	Py_XDECREF(mv_ref_arr);
    	Py_XDECREF(depth_qp_arr);
    	Py_XDECREF(residual_Y);
    	return residual_UV;
    }else{
    	Py_XDECREF(mv_ref_arr);
    	Py_XDECREF(depth_qp_arr);
    	Py_XDECREF(residual_Y);
    	Py_XDECREF(residual_UV);
    	return Py_None;
    }
    
}


static PyMethodDef CoviexinfoMethods[] = {
    {"extract",extract,METH_VARARGS,"extract compressed info from video."},
    {"get_num_frames",  get_num_frames, METH_VARARGS, "Getting number of frames in a video."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef Coviexinfomodule = {
    PyModuleDef_HEAD_INIT,
    "coviexinfo",   /* name of module */
    NULL,       /* module documentation, may be NULL */
    -1,         /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CoviexinfoMethods
};


PyMODINIT_FUNC PyInit_coviexinfo(void)
{
    PyObject *m;

    m = PyModule_Create(&Coviexinfomodule);
    if (m == NULL)
        return NULL;

    /* IMPORTANT: this must be called */
    import_array();

    CoviexinfoError = PyErr_NewException("coviexinfo.error", NULL, NULL);
    Py_INCREF(CoviexinfoError);
    PyModule_AddObject(m, "error", CoviexinfoError);
    return m;
}


int main(int argc, char *argv[])
{
    av_log_set_level(AV_LOG_QUIET);

    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("coviexinfo", PyInit_coviexinfo);

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    PyMem_RawFree(program);
    return 0;
}