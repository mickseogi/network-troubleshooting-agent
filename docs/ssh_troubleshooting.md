# SSH Troubleshooting

SSH 접속이 안 되는 경우에는 네트워크 연결, SSH 서비스 상태, 포트 22번, 방화벽 설정을 순서대로 확인해야 한다.

## Common Symptoms

- SSH 접속 시간이 초과된다.
- Connection refused 오류가 발생한다.
- 서버 IP로 ping은 되지만 SSH 접속이 안 된다.
- 특정 네트워크에서만 SSH 접속이 실패한다.

## Possible Causes

- 서버가 꺼져 있거나 네트워크에 연결되어 있지 않다.
- SSH 서비스가 실행 중이지 않다.
- 서버가 22번 포트를 열고 있지 않다.
- 방화벽에서 SSH 포트를 차단하고 있다.
- 클라이언트가 잘못된 IP 또는 포트로 접속하고 있다.

## Recommended Commands

- ping <server_ip>
- systemctl status sshd
- ss -tulnp | grep :22
- firewall-cmd --list-all