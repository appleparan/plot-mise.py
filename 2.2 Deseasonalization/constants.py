from pytz import timezone

SEOUL_CODES = [
    111121,111123,111131,111141,111142,
    111151,111152,111161,111171,111181,
    111191,111201,111212,111221,111231,
    111241,111251,111261,111262,111273,
    111274,111281,111291,111301,111311]
SEOUL_NAMES = [
    "중구","종로구","용산구","광진구","성동구",
    "중랑구","동대문구","성북구","도봉구","은평구",
    "서대문구","마포구","강서구","구로구","영등포구",
    "동작구","관악구","강남구","서초구","송파구",
    "강동구","금천구","강북구","양천구","노원구"]

# used in file name
SEOUL_NAMES_ENGDICT = {
    "중구": "jung",
    "종로구": "jongno",
    "용산구": "yongsan",
    "광진구": "gwangjin",
    "성동구": "seongdong",
    "중랑구": "jungnang",
    "동대문구": "dongdaemun",
    "성북구": "seongbuk",
    "도봉구": "dobong",
    "은평구": "eunpyeong",
    "서대문구": "seodaemun",
    "마포구": "mapo",
    "강서구": "gangseo",
    "구로구": "guro",
    "영등포구": "yeongdeungpo",
    "동작구": "dongjak",
    "관악구": "gwanak",
    "강남구": "gangnam",
    "서초구": "seocho",
    "송파구": "songpa",
    "강동구": "gangdong",
    "금천구": "geumcheon",
    "강북구": "gangbuk",
    "양천구": "yangcheon",
    "노원구": "nowon"}


SEOUL_STATIONS = dict(zip(SEOUL_NAMES, SEOUL_CODES))
SEOULTZ = timezone('Asia/Seoul')
