# データ作成プログラムへようこそ！

## Dockerの起動方法
1. `docker-compose up -d` (コンテナの起動)
2. `docker-compose exec research bash` (bashを起動)
3. これでPythonのプログラムを実行することが出来ます！
### コンテナを止める場合
4. `Ctrl+D` (bashから抜ける)
5. `docker-compose stop research` (コンテナを停止させる)

## プログラムの使用方法
- **上記の`2`までを行った後**
- `python3 make_data.py` (ミラー&観測したデータの作成)
- `python3 make_mirror_data.py` (ミラーの作成)
- `python3 make_detec_data.py` (観測したデータの作成 ※Json内にフォルダ名を記入する必要があります。)