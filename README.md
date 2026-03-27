# Lidar Spider #
---
**Maintainer**: TaoLiangjie  
**Affiliation**: HIT  
**Contact**: 3190312883@qq.com  

---
# 一.注意
1.本仓库没有 extended_legged_gym/legged_gym/resources/* 与 extended_legged_gym/LidarSensor/LidarSensor/resources/* 之下的文件，请自行复制过来。  
2.extended_legged_gym/legged_gym/legged_gym/scripts/play.py第47-48行与具体任务的参数表冲突，注释掉更好。  
3.


# 二.git使用指南  
## 1.链接远程仓库并同步分支状态
```bash
git remote add origin git@github.com:CZY412/OmniPerception.git
git remote -v # 检查状态
git fetch origin # 同步远程分支信息
```
## 2.拉取远程分支
(还没本地分支）
```bash
git switch -c <新分支名1> origin/<新分支名2> # <新分支名1>为本地名称，origin/<新分支名2>为远程名称，-c是新建再拉取的意思
```
例如：git switch -c TLJ origin/TLJ

## 3.修改提交信息
```bash
git log --oneline -n 5 # 查看提交历史
git reset --soft HEAD~1 # 软撤回，已经修改的部分保留为已暂存状态
git reset --mixed HEAD~1 # 或者取消暂存，改为未暂存状态

git add . 
git commit -m ""
git push --force-with-lease origin 分支名 # 强制推送，用本地历史替换远端
```

## 4.基于当前分支新建分支
```bash
git checkout -b temp
git push -u origin temp
```

## 5.重命名分支
```bash
git branch -m temp
```

## 6.删除分支
```bash
git branch -d temp
```

## 7.合并分支
```bash
git merge --no-ff temp # 把temp合并到当前分支
```