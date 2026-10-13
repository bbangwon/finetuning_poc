"""가상 사내 매뉴얼(data/manual.md)을 기반으로 파인튜닝용 Train/Test 데이터셋(JSONL)을 생성하는 스크립트."""

import json
from pathlib import Path

# Gemma 및 일반 Instruction 모델에서 널리 쓰이는 표준 포맷 정의
# chat template 스타일 (role: system, user, assistant) 및 SFT 기본 (instruction, input, output)

SYSTEM_PROMPT = (
    "당신은 넥스트웨이브 코퍼레이션(NextWave Corp.)의 사내 인사/총무 안내 AI 어시스턴트 '웨이브봇(WaveBot)'입니다. "
    "사내 복무 및 운영 규정집에 근거하여 직원들의 문의에 정확하고 친절하게 답변하세요."
)

# 학습용 데이터셋 (다양한 말투, 질문 변형, 상황 부여)
TRAIN_DATA = [
    # 1. 근무 시간 및 유연근무
    {
        "instruction": "아침 몇 시까지 출근해야 하나요?",
        "output": "넥스트웨이브는 시차출퇴근제를 운영하고 있어 오전 8시부터 10시 사이에 자율적으로 출근하시면 됩니다. 출근 시간 기준 휴게시간 1시간을 제외하고 8시간 근무 후 퇴근하시면 됩니다. 단, 전원 집중 협업 시간인 코어 타임(14:00~16:00)은 반드시 지켜주셔야 합니다."
    },
    {
        "instruction": "코어 타임이 몇 시부터 몇 시까지죠? 그 시간에 병원 잠깐 다녀와도 되나요?",
        "output": "사내 코어 타임(Core Time)은 14:00부터 16:00까지입니다. 이 시간은 전원 협업 및 회의 집중 시간으로 원칙적으로 외출이나 자리비움이 제한됩니다. 불가피한 외출이나 병원 방문이 필요한 경우 반반차(2시간) 등의 휴가를 사용하시길 권장합니다."
    },
    {
        "instruction": "9시 30분에 출근하면 몇 시에 퇴근하면 되나요?",
        "output": "오전 9시 30분에 출근하셨다면 점심 휴게시간 1시간을 포함하여 총 9시간 후인 오후 6시 30분(18:30)에 퇴근하시면 됩니다."
    },

    # 2. 재택근무 규정
    {
        "instruction": "재택근무는 일주일에 며칠까지 쓸 수 있나요?",
        "output": "재택근무는 주 2회(월 최대 8회) 자율적으로 사용하실 수 있습니다. 단, 화요일과 목요일은 전사 코어데이(Core Day)로 지정되어 있어 오프라인 오피스로 반드시 출근하셔야 합니다."
    },
    {
        "instruction": "화요일에 집에서 일하고 싶은데 재택 신청 되나요?",
        "output": "아니요, 화요일과 목요일은 전사 코어데이(Core Day)로 전원 오피스 출근이 필수입니다. 따라서 화요일에는 재택근무를 신청하실 수 없으며, 월/수/금 요일 중에 주 2회를 활용해 주시기 바랍니다."
    },
    {
        "instruction": "재택근무 신청은 당일 아침에 해도 되나요?",
        "output": "재택근무는 당일 신청이 불가합니다. 근무 전일 오후 6시까지 사내 포털인 WavePortal을 통해 신청하고 팀장 승인을 완료하셔야 합니다."
    },
    {
        "instruction": "재택근무 중에 잠깐 은행 다녀오려고 하는데 어떻게 해야 하나요?",
        "output": "재택근무 중에는 Slack 상태를 '재택근무'로 설정하셔야 하며, 15분 이상 자리를 비우실 경우 반드시 상태를 '자리비움'으로 변경해 팀원들에게 공유해 주셔야 합니다."
    },

    # 3. 휴가 및 병가
    {
        "instruction": "반반차 제도가 있나요? 몇 시간 단위인가요?",
        "output": "네, 넥스트웨이브는 2시간 단위로 사용할 수 있는 반반차 제도와 4시간 단위의 반차 제도를 모두 지원하고 있습니다."
    },
    {
        "instruction": "아침에 급한 일이 생겨서 당일 연차를 쓰고 싶어요. 언제까지 신청해야 하죠?",
        "output": "당일 긴급 연차 또는 반차는 당일 오전 8시 30분까지 사내 포털(WavePortal)에서 신청하시고, 소속 부서 Slack 채널에 미리 공지해 주시면 됩니다."
    },
    {
        "instruction": "유급 병가는 1년에 며칠까지 쓸 수 있고 증빙 서류가 필요한가요?",
        "output": "유급 병가는 연간 최대 5일까지 사용 가능합니다. 사용 시 의사 진단서 또는 진료확인서 제출이 필수이며, 3일 이상 연속으로 병가를 사용하는 경우에는 전문의 진단서를 첨부하셔야 합니다."
    },
    {
        "instruction": "입사한 지 만 3년이 되었는데 안식휴가가 나오나요?",
        "output": "네, 축하드립니다! 만 3년 근속 시 리프레시 휴가로 5일의 유급휴가와 함께 휴가비 50만 원이 지원됩니다. (만 5년 근속 시에는 10일 유급휴가와 100만 원 지원)"
    },

    # 4. 식대 및 야근 지원
    {
        "instruction": "점심 식대는 따로 법인카드로 결제하면 되나요?",
        "output": "점심 식대는 법인카드 결제 대상이 아니며, 매월 급여에 비과세 식대 월 20만 원이 일괄 포함되어 지급됩니다."
    },
    {
        "instruction": "야근 식대 한도와 인정 기준 시간 알려줘.",
        "output": "야근 식대는 오후 8시(20:00) 이후까지 연장 근무할 경우 지원되며, 1인당 한도는 배달팁 포함 최대 15,000원입니다. 결제는 법인카드 사용이 원칙입니다."
    },
    {
        "instruction": "야근하고 밤 10시 넘어서 퇴근하는데 택시비 지원되나요?",
        "output": "네, 오후 10시(22:00) 이후 퇴근 시에는 카카오 T 비즈니스를 통해 야근 교통비(택시비)가 전액 실비로 지원됩니다."
    },
    {
        "instruction": "불가피하게 개인카드로 야근 식대를 긁었는데 정산 어떻게 하나요?",
        "output": "개인카드로 결제하신 경우, 영수증을 첨부하여 익월 5일까지 WavePortal을 통해 비용 정산 신청을 완료해 주시면 급여 계좌로 환급됩니다."
    },

    # 5. 장비 및 복지
    {
        "instruction": "개발자인데 맥북 프로 지원 금액 한도가 얼마인가요?",
        "output": "개발/엔지니어링 직군은 3년 주기로 'W-Gear' 제도를 통해 최대 350만 원 상당의 MacBook Pro 또는 워크스테이션이 지급됩니다."
    },
    {
        "instruction": "비개발 직군 노트북 지원 기준은 어떻게 되나요?",
        "output": "비개발 직군의 경우 3년 주기로 최대 200만 원 상당의 업무용 노트북이 지급됩니다."
    },
    {
        "instruction": "3년 동안 쓰던 업무용 맥북을 퇴사하거나 교체할 때 개인이 가질 수 있나요?",
        "output": "네, 3년 감가상각이 종료된 장비는 희망 직원에 한해 잔존가 5만 원을 지불하고 개인 소유로 양도받으실 수 있습니다."
    },
    {
        "instruction": "자기계발비로 헬스장 등록해도 지원되나요?",
        "output": "아니요, 운동 시설 및 헬스장 등록은 자기계발비 지원 대상에서 제외됩니다. 대신 사내 피트니스 센터를 무료로 이용하실 수 있습니다. (자기계발비는 연간 120만 원 한도로 도서 구매, 온라인 강의, 자격증 응시료에 한해 지원됩니다.)"
    },

    # 6. 보안 규정
    {
        "instruction": "업무 코드를 ChatGPT 웹사이트에 넣어서 오류 찾아봐도 되나요?",
        "output": "아니요, 절대 금지되어 있습니다. 사내 보안 규정에 따라 사내 소스코드 및 고객 데이터를 외부 공개형 AI(ChatGPT 등)에 직접 입력하는 것은 엄격히 금지됩니다. 반드시 회사에서 승인된 사내 전용 AI 및 프라이빗 엔드포인트만 사용하셔야 합니다."
    }
]

# 평가/검증용 데이터셋 (학습 문장과 표현이 다르지만 같은 사규 지식을 묻는 10가지 테스트 케이스)
TEST_DATA = [
    {
        "id": "Q1_work_hours",
        "instruction": "출근을 10시에 해도 사규 위반이 아닌가요?",
        "expected_answer": "오전 8시~10시 사이 시차출퇴근제이므로 위반이 아니며, 8시간 근무 후 오후 7시 퇴근."
    },
    {
        "id": "Q2_core_day",
        "instruction": "이번 주 목요일에 집에서 일할 수 있나요?",
        "expected_answer": "목요일은 전사 코어데이(Core Day)이므로 재택 불가, 오피스 출근 필수."
    },
    {
        "id": "Q3_remote_apply",
        "instruction": "내일 재택근무 하려면 오늘 언제까지 신청해야 하죠?",
        "expected_answer": "전일 오후 6시까지 WavePortal에서 신청 및 팀장 승인 완료 필수."
    },
    {
        "id": "Q4_emergency_leave",
        "instruction": "오늘 늦잠을 자서 반반차 쓰려고 하는데 몇 시까지 올려야 해요?",
        "expected_answer": "당일 오전 8시 30분까지 WavePortal 신청 및 부서 슬랙 공지."
    },
    {
        "id": "Q5_dinner_time_limit",
        "instruction": "저녁 7시 30분에 퇴근하면서 야근 식대 청구 가능한가요?",
        "expected_answer": "불가. 야근 식대는 오후 8시(20:00) 이후까지 근무해야 지원 대상."
    },
    {
        "id": "Q6_dinner_price_limit",
        "instruction": "야근 저녁 밥값 최대 얼마까지 결제할 수 있나요?",
        "expected_answer": "1인당 최대 15,000원 (배달팁 포함)."
    },
    {
        "id": "Q7_taxi_support",
        "instruction": "밤 10시 30분에 야근 끝나고 택시 타고 가려는데 영수증 어떻게 처리하나요?",
        "expected_answer": "오후 10시 이후이므로 카카오 T 비즈니스로 전액 실비 지원 (자동 처리)."
    },
    {
        "id": "Q8_laptop_limit_dev",
        "instruction": "엔지니어 신규 입사자 노트북 예산 한도가 얼마인가요?",
        "expected_answer": "개발/엔지니어링 직군은 최대 350만 원 상당."
    },
    {
        "id": "Q9_fitness_allowance",
        "instruction": "필라테스 학원 결제하고 자기계발비 신청해도 되죠?",
        "expected_answer": "불가. 운동 시설은 자기계발비 제외(사내 피트니스 무료 이용), 도서/강의/자격증만 연 120만원 지원."
    },
    {
        "id": "Q10_external_ai",
        "instruction": "버그 잡기 힘들어서 우리 서비스 코드를 퍼플렉시티나 챗지피티에 질문해봐도 돼요?",
        "expected_answer": "절대 불가. 외부 공개형 AI에 소스코드 입력 금지, 승인된 사내 전용 AI만 사용 가능."
    }
]


def export_datasets():
    data_dir = Path(__file__).parent
    
    # 1. train.jsonl과 valid.jsonl 분리 (mlx-lm 규격)
    # 20개 중 16개 train, 4개 valid
    train_records = TRAIN_DATA[:16]
    valid_records = TRAIN_DATA[16:]

    def to_chat_format(item):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]}
            ]
        }

    train_file = data_dir / "train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_records:
            f.write(json.dumps(to_chat_format(item), ensure_ascii=False) + "\n")

    valid_file = data_dir / "valid.jsonl"
    with open(valid_file, "w", encoding="utf-8") as f:
        for item in valid_records:
            f.write(json.dumps(to_chat_format(item), ensure_ascii=False) + "\n")

    # 최종 비교 평가용 10개 질문 (test.jsonl 또는 test_questions.jsonl)
    test_file = data_dir / "test_questions.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for item in TEST_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # mlx-lm용 test.jsonl도 valid와 동일하게 chat format으로 생성
    mlx_test_file = data_dir / "test.jsonl"
    with open(mlx_test_file, "w", encoding="utf-8") as f:
        for item in valid_records:
            f.write(json.dumps(to_chat_format(item), ensure_ascii=False) + "\n")

    print(f"✅ 학습 데이터: {train_file} ({len(train_records)}건)")
    print(f"✅ 검증 데이터: {valid_file} ({len(valid_records)}건)")
    print(f"✅ 비교 평가용 질문 데이터: {test_file} ({len(TEST_DATA)}건)")


if __name__ == "__main__":
    export_datasets()
