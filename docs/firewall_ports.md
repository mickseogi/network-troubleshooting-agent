# Firewall and Port Troubleshooting

방화벽 또는 포트 문제는 네트워크 연결은 가능하지만 특정 서비스 접속만 실패할 때 의심할 수 있다.

## Common Symptoms

- ping은 되지만 특정 서비스 접속이 안 된다.
- SSH, HTTP, MySQL 같은 특정 포트 접속만 실패한다.
- Connection refused 또는 timeout 오류가 발생한다.
- 서버 내부에서는 접속되지만 외부에서는 접속되지 않는다.

## Possible Causes

- 서버 방화벽이 포트를 차단하고 있다.
- 클라우드 보안 그룹 또는 네트워크 ACL이 차단하고 있다.
- 서비스가 해당 포트에서 실행 중이지 않다.
- 포트 포워딩이 잘못 설정되어 있다.
- 중간 네트워크 장비에서 포트를 차단하고 있다.

## Recommended Commands

- firewall-cmd --list-all
- ss -tulnp
- netstat -ano
- telnet <server_ip> <port>