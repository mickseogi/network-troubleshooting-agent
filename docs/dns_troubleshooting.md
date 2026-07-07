# DNS Troubleshooting

DNS 문제는 IP 주소로는 통신이 되지만 도메인 이름으로는 접속이 안 되는 경우에 의심할 수 있다.

## Common Symptoms

- ping 8.8.8.8은 되지만 google.com 접속은 안 된다.
- 웹사이트 주소를 입력하면 서버를 찾을 수 없다는 오류가 발생한다.
- 특정 도메인만 접속되지 않는다.

## Possible Causes

- DNS 서버 주소가 잘못 설정되어 있다.
- DNS 서버가 응답하지 않는다.
- /etc/resolv.conf 설정이 잘못되어 있다.
- 사내 DNS 또는 공유기 DNS 캐시 문제가 있다.
- 방화벽이 DNS 요청을 차단하고 있다.

## Recommended Commands

- nslookup google.com
- ping 8.8.8.8
- cat /etc/resolv.conf
- systemd-resolve --status