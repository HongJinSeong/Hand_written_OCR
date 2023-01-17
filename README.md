# Hand_written_OCR

OCR 첫 참가

최종 accuracy 77.7% 마무리 ( 모델 RCNN - naver git 참조하여 약간 수정 진행한 것) https://github.com/clovaai/deep-text-recognition-benchmark

CNN으로 feature 추출이후에 transformer 붙인 것이 성능이 좋아보였으나 자원이 모자라서 시도 해보진 못함 ( 69% )

전처리와 다른부분을 많이 조정해보았으나 성능에 약간의 차이는 있었으나 큰폭의 변화는 없었고 63% ==> 77.9%로 상승하는데 가장 많이 기여한 부분은 sequence length 조정이였음

sequence length를 짧게 가져갔을때는 (6) 수렴은 빨리하였으나 성능이 좋지 못했고 길게 가져갔을 때 (38) 일 때는 수렴을 빨리 하지 못하였으나 성능은 향상 된것으로 볼 때
sequence length를 길게 가져갈수록 (OCR 에서 sequence length는 wdith에 해당하는 feature 갯수) 분간하기 힘든 단어들도 더 잘 구분하는 것을 알수 있었음

다음에 참여할 기회가 되면 적정 sequence length부터 찾는거부터 시작하는 것이 좋을 것으로 보이고 

다음으로 변화 줄 부분은 CNN feature 추출하는 부분에 axial attention을 적용해보거나 RNN을 transformer 로 적용해보는 것이 좋을것으로 보임.

아예 새로운 모델을 시도한다면 Easter2.0: Improving convolutional models for handwritten text recognition ( 1-D CNN 기반 ) 을 참조하여 시도해보는 것도 좋을것으로 보임
단점은 Easter2.0는 pytorch가 아닌 tensorflow로 구현되어 있음
