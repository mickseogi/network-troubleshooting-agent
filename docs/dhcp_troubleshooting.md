# DHCP Troubleshooting

DHCP 문제가 발생하면 클라이언트가 IP 주소, 게이트웨이, DNS 정보를 자동으로 받지 못할 수 있다.

## Common Symptoms

- IP 주소가 169.254.x.x 대역으로 설정된다.
- 네트워크에 연결되어 있지만 인터넷이 되지 않는다.
- 클라이언트가 IP 주소를 받지 못한다.
- 같은 네트워크의 다른 장비는 정상적으로 동작한다.

## Possible Causes

- DHCP 서버가 동작하지 않는다.
- DHCP IP 풀에 남은 주소가 없다.
- 클라이언트와 DHCP 서버가 같은 네트워크에 있지 않다.
- VLAN 또는 라우터 설정 문제로 DHCP Discover 패킷이 전달되지 않는다.
- 잘못된 DHCP 서버가 응답하고 있다.

## Recommended Commands

- ipconfig /all
- ip addr
- nmcli dev show
- journalctl -u NetworkManager --no-pager