package ro.ubb.ai.javaserver.security;

import ro.ubb.ai.javaserver.enums.Role;
import lombok.Getter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.User; // Spring's User

import java.util.Collection;

@Getter
public class UserPrincipal extends User { // Extends Spring's User for convenience
    private final Long id;
    private final Role domainRole; // Store your domain Role enum

    public UserPrincipal(Long id, String username, String password,
                         Collection<? extends GrantedAuthority> authorities, Role domainRole) {
        super(username, password, authorities);
        this.id = id;
        this.domainRole = domainRole;
    }

    // You can add more getters for other user-specific info if needed
}
