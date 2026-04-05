如果只让我给你 **一个最稳妥、最适合“海外 VPS + Windows 管理 + 手机使用”** 的方案，我建议你用：

**Ubuntu 24.04/22.04 海外 VPS + WireGuard + wg-easy（Web 管理面板）**

原因很直接：WireGuard 本身就是以“简单、现代、性能高”为目标设计的，官方明确把它定位成比 OpenVPN 更简单、更轻量、性能也更好的通用 VPN；而 wg-easy 在它外面加了一层很省事的 Web UI，可以直接生成客户端、显示手机扫码二维码、导出 Windows 配置文件。对你这种“Windows 上管理、手机上连”的场景，非常合适。([WireGuard][1])

先说结论上的取舍：

* **最佳综合方案**：WireGuard + wg-easy
* **最傻瓜、点点点最多的方案**：Outline
* **最省配置、但要接受账号体系**：Tailscale Exit Node
* **不推荐优先选**：OpenVPN，能用，但更重、更老派。([Outline][2])

---

## 你需要准备什么

1. 一台**海外 VPS**，系统建议选 **Ubuntu 24.04 LTS 或 22.04 LTS**。Docker 官方当前支持这两个版本；wg-easy 也要求主机有**公网 IP 或域名**，并支持 x86_64/arm64。([Docker Documentation][3])

2. Windows 电脑一台。
   你只需要能 SSH 进服务器。最简单就是用 **Windows Terminal / PowerShell 里的 `ssh`**；没有的话装 PuTTY 也行。

3. 手机安装 **WireGuard App**。Windows 端也装 **WireGuard 官方客户端**。WireGuard 官方提供 Windows 安装器，Android 端官方提供 Play 版/APK。([download.wireguard.com][4])

---

## 为什么我推荐这一套，而不是别的

wg-easy 官方特性里就写得很清楚：它是 **WireGuard + Web UI 一体化**，支持创建/删除客户端、显示二维码、下载配置文件、看连接状态。对手机来说，最方便的地方就是**扫码即连**；对 Windows 来说，就是**导入 `.conf` 即可**。([GitHub][5])

另外，Docker 官方现在明确建议生产环境优先走 **apt 仓库安装 Docker Engine**，而不是一把梭安装脚本；同时 Docker 也提醒，容器映射出来的端口可能绕过你原本的 ufw 规则，所以管理面板最好不要裸露在公网。([Docker Documentation][3])

---

# 详细教程：按这个做就行

下面我按 **“没有域名，也能做”** 的思路教你。
核心思路是：

* VPN 端口 `51820/udp` 正常对外开放
* **管理面板 `51821/tcp` 不直接暴露公网**
* 你在 Windows 上通过 **SSH 本地端口转发** 来打开面板
* 这样比直接公网开管理后台安全得多

---

## 第 1 步：买海外 VPS

配置不需要很高。个人自用，**1 vCPU / 1 GB RAM / 20 GB SSD** 就够了。地区尽量选离你近的海外机房，例如新加坡、日本、韩国这类，通常延迟更低。这部分是经验判断，但你真正需要满足的硬条件是：**Ubuntu 22.04/24.04 + 公网 IPv4**。([Docker Documentation][3])

创建服务器时建议：

* 系统：Ubuntu 24.04 LTS
* 开放公网 IP
* 记录好：`服务器IP`、`root 密码` 或 SSH key

---

## 第 2 步：用 Windows 连上服务器

在 Windows 的 PowerShell 里执行：

```powershell
ssh root@你的服务器IP
```

第一次连接会让你确认指纹，输入 `yes`。
然后输入密码即可。

登录后先更新系统：

```bash
sudo apt update && sudo apt upgrade -y
```

---

## 第 3 步：按官方方式安装 Docker

Docker 官方在 Ubuntu 上推荐先加官方 apt 仓库，再安装 `docker-ce`、`docker-compose-plugin` 等组件。照下面原样执行即可。([Docker Documentation][3])

```bash
sudo apt remove -y docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc || true

sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

安装完检查：

```bash
sudo systemctl status docker --no-pager
sudo docker run hello-world
```

如果看到 `hello-world` 成功输出，说明 Docker 已经正常。([Docker Documentation][3])

---

## 第 4 步：下载 wg-easy 官方 compose 文件

wg-easy 官方文档给出的最基本流程是：创建目录、下载官方 `docker-compose.yml`、然后 `docker compose up -d`。官方默认对外开放 `51820/udp` 和 `51821/tcp`，其中 `51820` 是 WireGuard 端口。([WG Easy][6])

执行：

```bash
sudo mkdir -p /etc/docker/containers/wg-easy
cd /etc/docker/containers/wg-easy
sudo curl -o docker-compose.yml https://raw.githubusercontent.com/wg-easy/wg-easy/master/docker-compose.yml
```

---

## 第 5 步：修改 compose，让管理面板不暴露公网

wg-easy 官方“无反代”说明里明确说了：**不加反向代理、直接明文开 Web UI 是不安全的**；只有在 Web UI **不对公网开放** 时才勉强可接受。([WG Easy][7])

所以我们改两处：

### 1）把管理面板只绑定到本机回环地址

编辑文件：

```bash
sudo nano /etc/docker/containers/wg-easy/docker-compose.yml
```

找到这部分（官方默认大概是这样）：

```yaml
ports:
  - "51820:51820/udp"
  - "51821:51821/tcp"
```

把第二行改成：

```yaml
  - "127.0.0.1:51821:51821/tcp"
```

这样管理后台只在服务器本机可见，公网打不开。

### 2）开启无反代模式

在 compose 里，官方文件已经给了注释掉的 `environment` 示例，其中包括：

```yaml
#environment:
#  - PORT=51821
#  - HOST=0.0.0.0
#  - INSECURE=false
```

把它改成：

```yaml
environment:
  - INSECURE=true
```

你最终至少要保证：

* `51820/udp` 对外
* `51821/tcp` 只绑定到 `127.0.0.1`
* `INSECURE=true`

这是因为 wg-easy 官方说得很明确：**不用反向代理时，Web UI 是不安全的，不应暴露到公网**。([GitHub][8])

---

## 第 6 步：启动 wg-easy

```bash
cd /etc/docker/containers/wg-easy
sudo docker compose up -d
sudo docker ps
```

如果容器状态是 `Up`，说明起来了。
以后更新也很简单：

```bash
cd /etc/docker/containers/wg-easy
sudo docker compose pull
sudo docker compose up -d
```

这也是 wg-easy 官方文档给的更新方式。([WG Easy][6])

---

## 第 7 步：放行 VPS 防火墙和云厂商安全组

wg-easy 官方文档明确要求你至少开放：

* **UDP 51820**（WireGuard）([WG Easy][6])

所以你要检查两层：

### 云厂商控制台安全组 / 防火墙

放行：

* `22/tcp`（SSH）
* `51820/udp`（WireGuard）

### 服务器内的 UFW（如果你启用了）

可以执行：

```bash
sudo ufw allow 22/tcp
sudo ufw allow 51820/udp
sudo ufw enable
sudo ufw status
```

注意，Docker 官方特别提醒：**Docker 暴露的端口可能绕过 ufw 规则**。所以我们前面才把 `51821` 直接绑到 `127.0.0.1`，这样最省心。([Docker Documentation][3])

---

## 第 8 步：从 Windows 打开管理面板

因为 `51821` 没有暴露公网，所以你不能直接浏览器访问服务器 IP。
正确方式是在 Windows 上做 **SSH 本地转发**：

在 PowerShell 里重新开一个窗口，执行：

```powershell
ssh -L 51821:127.0.0.1:51821 root@你的服务器IP
```

这条命令保持不要关。
然后在你本机浏览器打开：

```text
http://127.0.0.1:51821
```

这时你看到的就是 wg-easy 管理后台。

---

## 第 9 步：首次初始化 wg-easy

wg-easy 的首次向导会让你创建管理员账户；社区教程里给出的流程是：创建 admin 账号，然后填写 **Host**。教程用的是域名示例；而 wg-easy 基本文档说明，部署前提是**域名或公网 IP 都可以**，所以你没有域名的话，这里直接填 **服务器公网 IP** 即可。([GitHub][9])

建议这样填：

* **Host**：你的服务器公网 IP
* **Port**：默认 `51820`
* 其他保持默认即可

---

## 第 10 步：创建手机和 Windows 客户端

在面板里新建两个客户端：

* `phone`
* `windows`

wg-easy 的核心价值就在这一步：它支持**显示二维码**和**下载配置文件**。([GitHub][5])

### 手机接入

手机安装 WireGuard App 后，在 wg-easy 后台点 `phone` 的二维码，用手机扫码导入。社区教程也明确写了 iOS/Android 直接用官方 WireGuard App 扫码。([GitHub][9])

### Windows 接入

Windows 安装官方 WireGuard 客户端，然后在 wg-easy 下载 `windows.conf`，导入进去即可。社区教程同样是这么做的。([GitHub][9])

---

## 第 11 步：连上后怎么验证是否成功

手机或 Windows 打开 WireGuard 连接后，做三件事：

1. 打开浏览器访问任意查 IP 网站，看出口 IP 是否变成你的 VPS IP
2. 测一下网页加载速度是否正常
3. 看 wg-easy 后台是否显示客户端在线、有没有 Tx/Rx 流量

wg-easy 本身就支持查看连接统计和流量图。([GitHub][5])

---

# 常见问题

## 1）手机能连上，但打不开网页

最常见是这几个原因：

* 云厂商安全组没放行 `51820/udp`
* VPS 本机防火墙没放行 `51820/udp`
* 容器没正常起来：

  ```bash
  sudo docker ps
  sudo docker logs wg-easy --tail 100
  ```

---

## 2）后台打不开

先确认你是不是用了这条 SSH 转发：

```powershell
ssh -L 51821:127.0.0.1:51821 root@你的服务器IP
```

然后浏览器访问的是：

```text
http://127.0.0.1:51821
```

不是服务器公网 IP，也不是 `https`。

---

## 3）某些网络下连不上

WireGuard 官方自己就说过，它**不以流量伪装为目标**，而且底层是 **UDP**。所以如果你所在网络对 UDP 或未知协议限制比较严，WireGuard 可能会不如别的方案好连。([WireGuard][10])

这种情况下有两个替代思路：

* **Outline**：最省事，Windows 上装 Manager，直接建服务、发 access key，手机装 Client 导入即可。Outline 官方就是这么设计的。([Outline][2])
* **Tailscale Exit Node**：如果你能接受每台设备登录同一个 Tailscale 网络，那它非常省配置。官方流程是 VPS 上开 IP forwarding，然后 `sudo tailscale set --advertise-exit-node`，再在后台批准；手机端在 Exit Node 里选择该服务器即可。([Tailscale][11])

---

# 我给你的最终建议

按你的需求，优先顺序我会这样排：

**第一选择：WireGuard + wg-easy**
适合：想要标准 VPN、速度快、手机扫码接入、Windows 也能方便管理。([WireGuard][1])

**第二选择：Outline**
适合：你就是想尽可能少敲命令、少折腾 Linux 配置。Outline 官方的 Windows Manager 就是为这种“一台桌面端管服务器、手机端直接连”设计的。([Outline][2])

**第三选择：Tailscale Exit Node**
适合：你接受账号体系，并且希望以后不只是“手机翻全局”，还想做多设备互联。([Tailscale][11])

如果你愿意，我下一条可以直接给你一份 **可复制粘贴的最终版 `docker-compose.yml`**，我会按“**没有域名、只用公网 IP、Windows SSH 转发管理**”的形式整理好。

[1]: https://www.wireguard.com/?utm_source=chatgpt.com "WireGuard: fast, modern, secure VPN tunnel"
[2]: https://getoutline.org/get-started/ "Outline"
[3]: https://docs.docker.com/engine/install/ubuntu/ "Ubuntu | Docker Docs"
[4]: https://download.wireguard.com/windows-client/?utm_source=chatgpt.com "WireGuard for Windows Downloads - Installation"
[5]: https://github.com/wg-easy/wg-easy "GitHub - wg-easy/wg-easy: The easiest way to run WireGuard VPN + Web-based Admin UI. · GitHub"
[6]: https://wg-easy.github.io/wg-easy/latest/examples/tutorials/basic-installation/ "Basic Installation - wg-easy"
[7]: https://wg-easy.github.io/wg-easy/latest/examples/tutorials/reverse-proxyless/ "No Reverse Proxy - wg-easy"
[8]: https://github.com/wg-easy/wg-easy/blob/master/docker-compose.yml "wg-easy/docker-compose.yml at master · wg-easy/wg-easy · GitHub"
[9]: https://github.com/hetzneronline/community-content/blob/master/tutorials/installing-wireguard-ui-using-docker-compose/01.en.md "community-content/tutorials/installing-wireguard-ui-using-docker-compose/01.en.md at master · hetzneronline/community-content · GitHub"
[10]: https://www.wireguard.com/known-limitations/?utm_source=chatgpt.com "Known Limitations"
[11]: https://tailscale.com/docs/features/exit-nodes/how-to/setup "Use exit nodes · Tailscale Docs"
