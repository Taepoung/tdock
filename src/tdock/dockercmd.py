"""tdock 이 docker 를 부르는 명령.

보통은 ["docker"]. docker 그룹에 없고 sudo 로 docker 만 허용된 서버에서는
시작할 때 ["sudo", "-n", "docker"] 로 바뀐다 (app.main).
"""

DOCKER = ["docker"]
