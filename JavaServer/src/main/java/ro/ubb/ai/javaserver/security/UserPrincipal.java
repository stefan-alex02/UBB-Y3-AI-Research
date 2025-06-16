package ro.ubb.ai.javaserver.security;

import lombok.Getter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.User;
import ro.ubb.ai.javaserver.enums.Role;

import java.util.Collection;

@Getter
public class UserPrincipal extends User {
    private final Long id;
    private final Role domainRole;

    public UserPrincipal(Long id, String username, String password,
                         Collection<? extends GrantedAuthority> authorities, Role domainRole) {
        super(username, password, authorities);
        this.id = id;
        this.domainRole = domainRole;
    }
}
