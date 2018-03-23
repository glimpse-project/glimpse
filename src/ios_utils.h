
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

char *
ios_util_get_documents_path(void);

char *
ios_util_get_resources_path(void);

struct ios_av_session;

struct ios_av_session *
ios_util_av_session_new(struct gm_logger *log);

#ifdef __cplusplus
}
#endif
