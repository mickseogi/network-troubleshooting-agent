# Gateway and Routing Troubleshooting

게이트웨이 또는 라우팅 문제는 같은 네트워크 내부 통신은 가능하지만 외부 네트워크나 인터넷 접속이 안 될 때 의심할 수 있다.

## Common Symptoms

- 같은 LAN 내부 장비로는 ping이 된다.
- 외부 IP로 ping이 안 된다.
- 기본 게이트웨이로 ping이 실패한다.
- 특정 네트워크 대역으로만 통신이 안 된다.

## Possible Causes

- 기본 게이트웨이가 설정되어 있지 않다.
- 게이트웨이 IP가 잘못 설정되어 있다.
- 라우팅 테이블에 잘못된 경로가 있다.
- 라우터 또는 L3 스위치 설정에 문제가 있다.
- VLAN 간 라우팅이 설정되어 있지 않다.

## Recommended Commands

- ip route
- route print
- ping <gateway_ip>
- traceroute 8.8.8.8