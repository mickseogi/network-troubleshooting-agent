# Network Infrastructure Troubleshooting

네트워크 인프라 문제는 스위치, 라우터, 케이블, 대역폭, 장비 과부하 등 여러 요소로 인해 발생할 수 있다.

## Common Symptoms

- 네트워크 속도가 느리다.
- 간헐적으로 연결이 끊긴다.
- 특정 구간을 지나면 통신이 실패한다.
- 같은 서비스라도 위치에 따라 접속 성공 여부가 다르다.

## Possible Causes

- 케이블 또는 물리 포트에 문제가 있다.
- 스위치 또는 라우터에 과부하가 있다.
- 네트워크 대역폭이 부족하다.
- 홉 수가 많거나 중간 장비에서 병목이 발생한다.
- QoS 또는 네트워크 정책 설정이 잘못되어 있다.

## Recommended Commands

- ping <target_ip>
- traceroute <target_ip>
- ip route
- netstat -ano
- Wireshark packet capture